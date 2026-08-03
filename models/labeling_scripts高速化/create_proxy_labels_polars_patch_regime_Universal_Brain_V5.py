# /workspace/models/create_proxy_labels_polars_patch_regime.py
# [フェーズ3: 最終ラベリングスクリプト - V5 双方向ラベリング仕様]

import sys
from pathlib import Path
import warnings
import argparse
import shutil
from dataclasses import dataclass, field
import logging
from typing import List, Dict, Any, Optional, Tuple
import polars as pl
from tqdm import tqdm
import re
import gc
import datetime as dt
import numpy as np
import calendar

try:
    from numba import njit, prange
    from numba.core.errors import NumbaPerformanceWarning

    warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)
    NUMBA_AVAILABLE = True
except ImportError:
    logging.warning(
        "Numba not found. Labeling performance will be significantly degraded."
    )
    NUMBA_AVAILABLE = False

# --- Project Path Setup ---
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# --- Blueprint Imports ---
from blueprint import (
    S5_NEUTRALIZED_ALPHA_SET, S2_FEATURES_VALIDATED, S6_LABELED_DATASET,
    S1_RAW_TICK_PARTITIONED, S1_PROCESSED, BARRIER_ATR_PERIOD, ATR_BASELINE_DAYS
)

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(processName)s - %(message)s",
)
warnings.filterwarnings("ignore", category=UserWarning, module="polars")
try:
    from polars.exceptions import PolarsUsePyarrowWarning

    warnings.filterwarnings("ignore", category=PolarsUsePyarrowWarning)
except ImportError:
    pass

# --- ▼▼▼ V5 双方向ラベリング ルール定義 ▼▼▼ ---

# 対象タイムフレームとATRの指定
TARGET_TIMEFRAMES = ["M3"]  # Optunaの結論: M3単体・ratio0.8・TD30min
ATR_PERIOD = BARRIER_ATR_PERIOD  # blueprintから取得
# =============================================================================
# ATR Ratio ゲート
# -----------------------------------------------------------------------------
# 0.0 にすると全バーが is_trigger=1 になる（＝ゲート撤廃）。
#
# 【撤廃の根拠 — 測定結果】
#   従来捨てていた atr_ratio<0.8 の帯に、6年安定の構造が存在することが判明した。
#   しかも取引中の帯より効果が大きい:
#       低ボラ帯 (<0.8) : 窓120分の効率比>=0.256 かつ |d|0.3-0.6 → t=15分で −0.2315 ATR
#                         t=−4.28、6年すべて同符号、χ²のp=0.366、損益分岐 ATR 1.04 USD
#       取引帯 (>=0.8)  : 窓360分の効率比>=0.107 かつ |d|1.3+   → t= 1分で −0.0816 ATR
#                         t=−4.78、6年すべて同符号、χ²のp=0.477、損益分岐 ATR 2.94 USD
#   USDに直しても（低ボラ帯はATR絶対値が小さいぶん割引いても）低ボラ帯の方が上。
#
#   さらに両者は【別々の構造】である:
#       低ボラ帯 = 小さな動き(0.3-0.6 ATR)が15分かけて戻る
#       取引帯   = 大きな動き(1.3+ ATR)が1分で戻る
#   混ぜて測ると互いに薄まるが、モデルは atr_ratio_M3 を特徴量として持つので
#   木が自分で場合分けできる。よってゲートで捨てる理由が無い。
#
#   母集団: 353,467 → 567,730 本（1.6倍）。弱い信号ほどデータ量が効く。
#
# 【下流への影響（確認済み）】
#   is_trigger はフィルタ条件としてしか使われておらず、全行が1になっても
#   壊れる箇所は無い。特徴量からは全スクリプトで除外されている。
#   Ax2 の scale_pos_weight は母集団から再計算されるので正しく追従する。
#   ※ BT 側は独立に min_atr_threshold を持つ。実行時 --min-atr 0.0 で揃えること。
# =============================================================================
ATR_RATIO_THRESHOLD = 0.0  # ATR Ratio閾値（0.0 = ゲート撤廃）

# timeframeごとの1日あたりバー数（ATR Ratio計算のbaseline_period算出に使用）
timeframe_bars_per_day = {
    "M0.5": 2880, "M1": 1440, "M3": 480, "M5": 288,
    "M8": 180, "M15": 96, "M30": 48, "H1": 24,
    "H4": 6, "H6": 4, "H12": 2, "D1": 1, "W1": 1, "MN": 1
}

# =============================================================================
# スプレッド — ラベリングでは 0 にする
# -----------------------------------------------------------------------------
# 実装上、SPREAD はバリア対を丸ごとずらす:
#     Long : PT = close + a·ATR + S 、SL = close − b·ATR + S
#     Short: PT = close − a·ATR − S 、SL = close + b·ATR − S
# これは「Askで入りBidで出る」往復コストを正しくモデル化しており、構造は正しい。
#
# 【だが 0 にする。理由 = ATR依存の偏りがラベルに入り、ボラ選抜の抜け道になるため】
#   無ドリフトでの label=1 の確率は
#       P(1) = b/(a+b) − S/((a+b)·ATR)
#   と、第2項に ATR が分母で入る。ATR が小さいほど label=1 が出にくい。
#   しかも対称バリアでは long も short も同じ向きに動く。
#
#   対称 a=b=1.4 / S=0.50 での実際の値:
#       ATR=1 → P(1)=0.321 、ATR=8 → P(1)=0.478   （振れ幅 15.6ポイント）
#   モデルは atr_value / atr_ratio を特徴量として持つので、
#   「ATRが大きいバーを選ぶ」だけで P(1) を 15.6pt 上げられてしまう。
#   long も short も同時に上がるため【方向を当てる必要がない】。
#   ＝ 時間切れを 0 に畳んでいたのと同じ構造の抜け道が、別ルートで開く。
#   （追っている方向のエッジは勝率にして 1〜3pt。抜け道の方が桁違いに大きい）
#
#   さらに対称化＋面積縮小でバリアが狭くなるため、この偏りは増幅される:
#       旧 1:5(面積5)      → 振れ幅  7.3pt
#       新 対称1.4(面積1.96) → 振れ幅 15.6pt
#
#   そして ATRゲート撤廃と真っ向からぶつかる。低ボラ帯を母集団に入れたのに、
#   「ATRが小さい→P(1)が低い」で系統的に 0 になり、モデルが
#   「低ボラを避けろ」と学習してしまう。＝ 外したゲートを内部で復活させる。
#   低ボラ帯こそ今回いちばん強い構造（−0.2315 ATR、6年安定）が出た場所である。
#
# 【コストはどこで見るか】
#   ・BT / 本番は spread_pips で実際に引く（実装済み）
#   ・「コストを賄えるか」は EV = 変位(ATR単位) × ATR(USD) − spread で判定できる。
#     これは取引の可否の話であって、モデルに学ばせる必要はない。
#   ・必要なら BT 側で ATR 下限や m2_proba 閾値で弾く。
#
# 【注意】ラベルは往復コストを含まない「理想の値動き」になる。
#   BT でコストが乗るぶん、BT の結果はラベル基準より必ず悪くなる。
#   それは正常であり、劣化ではない。
#
# 参考: 実測スプレッドは 0.240 USD（チャートの bid/ask を複数時点で確認）。
#       BT は spread_pips=36 × value_per_pip=1.0 → 0.360 USD 相当。
# =============================================================================
SPREAD = 0.0

# =============================================================================
# バリア幾何 — 対称化 + TD延長
# -----------------------------------------------------------------------------
# 【なぜ 1:5 / TD30 から変えるか — 測定で判明した2つの穴】
#
#   穴1「両方=1」: 非対称(1:5)だと、高ボラのバーで long も short も label=1 になる。
#       PT が +1 ATR と近いので、大きく振れれば両側とも先に PT に触れてしまう。
#       実測(無ドリフトMC): ボラ標準で18%、ボラ2倍で52%が「両方=1」。
#       → ラベルに方向情報が入らず、モデルは「よく動くか」を当てるだけで正解できる。
#       → 対称(a=b)にすると long=1 ⟺ short=0 が数学的に保証され、この穴は消える。
#          面積を縮めても非対称のままでは消えない（ボラ2倍で30%以上残る）。
#
#   穴2「時間切れ→0」: 決着しなかった玉を 0 に畳むため、
#       「時間内に決着するか(=よく動くか)」を当てるだけで 1/0 が分離できてしまう。
#       現行の時間切れ率 31.5%。
#       → 実測でも M2 は方向でなくボラを選んでいた（|ΔX|比 TOP/REST が 1.3〜1.45、
#          t統計量 19〜21。一方で方向の選抜効果は |t|<1.8 でゼロ）。
#       → TD を E[τ] の4倍以上にすれば時間切れが数%に落ち、この穴も消える。
#
#   ★2つは別々の穴。対称化だけでも、面積縮小だけでも塞がらない。両方必要。
#
# 【なぜ 1.4 / TD60 か】
#   面積 = pt×sl = 1.96。E[τ] = 面積 × 7.50分 = 14.7分。
#     （7.50分/面積1 は実測較正値。Parkinson理論7.64と−2%で一致）
#   効率比ルールの効果は t=15分でピーク（−0.152 ATR）なので、そこで決着させる。
#   TD=60分 は E[τ] の約4倍。無ドリフトMCでの時間切れ率は 1.6%（現行31.5%）。
#
#   ※ BT 側も pt=sl=1.4 / td=60 に合わせること。ずれると評価が食い違う。
# =============================================================================

# ロング用ルール
RULE_LONG = {
    "pt_mult": 1.4,   # pt_multiplier_long
    "sl_mult": 1.4,   # sl_multiplier_long
    "td": "60m",      # td_minutes_long: 60
}

# ショート用ルール
RULE_SHORT = {
    "pt_mult": 1.4,   # pt_multiplier_short
    "sl_mult": 1.4,   # sl_multiplier_short
    "td": "60m",      # td_minutes_short: 60
}

# =============================================================================
# 追加特徴量: 効率比（Efficiency Ratio）と 符号つき1バー変位
# -----------------------------------------------------------------------------
#   eff_ratio_K_{tf} = |close[t] − close[t−K]| / Σ|close[i] − close[i−1]|  （直近K本）
#       1 に近い = 直近K本を一方向に走った（その場がトレンド）
#       0 に近い = 行って戻った（その場がレンジ）
#
#   d_atr_{tf} = (close − open) / ATR      ← 符号つきの1バー変位
#       M3バー [L, L+180) では open=P(L)、close=P(L+180) なので、これは
#       測定で使ってきた d = P(L+180) − P(L) を ATR で正規化したものと同一。
#
# 【なぜ足すか — 測定で判明した構造（モデル非依存・6年安定）】
#   効率比が高い（＝すでに一方向に走った）バーの直後に大きめの1本が出ると、
#   その後は【逆行】する。行き過ぎの解消。
#       M3・窓60分・|d|0.6-0.9・ER>=0.356 → t=15分で −0.152 ATR (t=−4.2)
#       6年すべて同符号、χ²のp=0.60。効率比で絞らないと |t|<1.1 で完全に消える。
#   窓は M3足でも M1足でも「実時間60〜80分」が最適だった（バーの刻みに依らない）。
#   効率比の分布は6年間ほぼ不変（>=0.40 の割合が14〜15%）なので絶対値で扱える。
#
# 【既存の1343特徴量では代用できないことを確認済み】
#   e1f_biomechanical_efficiency は分子が経路長(Σ|vel|)で正味変位を使わず、
#     効率比との順位相関は 0.21〜0.33。名前が近いだけの別物。
#   e1c_adx_21 は順位相関 0.45。半分程度しか情報を共有しない。
#   e1c_kama_* は内部で効率比を使うが、出力は平滑価格の乖離率で効率比は消える。
#   符号つきの (close−open)/ATR も存在しない（e1d_body_size_atr は abs() 付きで
#     符号が落ちている）。よって効率比・符号つきdとも新規に必要。
#
# 【閾値は書かない】
#   測定で見つけた 0.356 等の閾値はここには入れない。値だけを特徴量として渡し、
#   閾値も |d| との組合せも木に学習させる。1点に固定すると他の情報と
#   組み合わせられなくなるため。
#
# 【なぜここで計算するか】
#   atr_ratio と同じ経路。S1_PROCESSED から作って S6 に列として出せば、
#   update_feature_list_v5.py がスキーマから自動で拾う。
#   エンジン（engine_1_*）も S2 も通し直す必要がない。
#   ★本番側（realtime_feature_engine.py）に同じ式を必ず入れること。
#     入れないと学習と本番で特徴量がずれる。
# -----------------------------------------------------------------------------
# 効率比の窓（バー本数）。実時間で 60分 と 75-80分 になるよう時間足ごとに設定する。
ER_WINDOWS_BY_TF = {
    "M0.5": (120, 150),
    "M1": (60, 80),
    "M3": (20, 25),
    "M5": (12, 15),
    "M8": (8, 10),
    "M15": (4, 5),
}
# --- ▲▲▲ 改造ここまで ▲▲▲ ---


# --- Configuration ---
@dataclass
class ProxyLabelConfig:
    """Config for creating a context-adaptive, dual-labeled subset for proxy model training."""

    input_dir: Path = S5_NEUTRALIZED_ALPHA_SET
    price_data_source_dir: Path = S1_PROCESSED  # tick/ATRはS1から直接取得（S2_FEATURES_VALIDATEDは不使用）
    output_dir: Path = S6_LABELED_DATASET

    filter_mode: str = "year"  # 'year', 'month', 'all'
    filter_year: Optional[int] = 2023
    filter_month: Optional[int] = None
    resume: bool = True
    execution_start_time: str = field(
        default_factory=lambda: dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )

    def get_filter_description(self) -> str:
        if self.filter_mode == "year" and self.filter_year is not None:
            return f"Year = {self.filter_year}"
        elif (
            self.filter_mode == "month"
            and self.filter_year is not None
            and self.filter_month is not None
        ):
            month_str = f"{self.filter_month:02d}"
            return f"Year/Month = {self.filter_year}/{month_str}"
        elif self.filter_mode == "all":
            return "All Time"
        else:
            return f"Invalid Filter ({self.filter_mode})"


# --- ▼▼▼ Numba 双方向走査エンジン ▼▼▼ ---
def _njit_if_available(func):
    if NUMBA_AVAILABLE:
        return njit(func, parallel=True, fastmath=True, cache=True)
    else:
        return func


@_njit_if_available
def _numba_find_hits_dual(
    bets_t0: np.ndarray,
    bets_t1_max_long: np.ndarray,
    bets_t1_max_short: np.ndarray,
    bets_pt_long: np.ndarray,
    bets_sl_long: np.ndarray,
    bets_pt_short: np.ndarray,
    bets_sl_short: np.ndarray,
    ticks_ts: np.ndarray,
    ticks_high: np.ndarray,
    ticks_low: np.ndarray,
    entry_offset_us: np.int64,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Numba JIT compiled function to find barrier hits for BOTH Long and Short simultaneously.

    [LOOKAHEAD-FIX §11.34.16] entry_offset_us:
      t0 はトリガー行のラベル L (バー開始時刻)。エントリーはバーの close
      (= L + ACTION_HORIZON_SEC、価格もその時点の close) で行われるため、
      バリア走査はエントリー時刻 (t0 + entry_offset_us) より後の tick から
      開始しなければならない。旧実装は t0 直後から走査しており、
      エントリー前 3 分間の tick (エントリー時点ではまだ取引が存在しない) が
      PT/SL 判定に混入していた → PT1.0 側に楽観バイアス。
      BT (エントリー後のバーのみで判定) / 本番 (発注後の価格のみ) と整合させる。
    """
    n_bets = len(bets_t0)
    n_ticks = len(ticks_ts)

    # ロング用とショート用の出力配列を準備（duration配列は削除）
    out_pt_long = np.zeros(n_bets, dtype=np.int64)
    out_sl_long = np.zeros(n_bets, dtype=np.int64)
    out_pt_short = np.zeros(n_bets, dtype=np.int64)
    out_sl_short = np.zeros(n_bets, dtype=np.int64)

    if n_ticks == 0:
        return out_pt_long, out_sl_long, out_pt_short, out_sl_short
    for i in prange(n_bets):
        t0 = bets_t0[i]

        # ロング用バリア・TD
        t1_l = bets_t1_max_long[i]
        pt_l = bets_pt_long[i]
        sl_l = bets_sl_long[i]

        # ショート用バリア・TD
        t1_s = bets_t1_max_short[i]
        pt_s = bets_pt_short[i]
        sl_s = bets_sl_short[i]

        # --- 修正前 ---
        # start_idx = np.searchsorted(ticks_ts, t0, side="left")
        # start_idx = np.searchsorted(ticks_ts, t0, side="right")  # 旧: ラベル直後から走査

        # --- [LOOKAHEAD-FIX §11.34.16] エントリー時刻 (t0 + offset) より後から走査 ---
        start_idx = np.searchsorted(ticks_ts, t0 + entry_offset_us, side="right")

        # ヒットしたタイムスタンプを記録する変数
        pt_l_found = np.int64(0)
        sl_l_found = np.int64(0)
        pt_s_found = np.int64(0)
        sl_s_found = np.int64(0)

        # 各方向の走査を継続するかどうかのフラグ
        long_active = True
        short_active = True

        for j in range(start_idx, n_ticks):
            tick_time = ticks_ts[j]
            tick_high = ticks_high[j]
            tick_low = ticks_low[j]

            # --- ロング判定 ---
            if long_active:
                if tick_time > t1_l:
                    long_active = False  # TD超過
                else:
                    if pt_l_found == 0 and tick_high >= pt_l:
                        pt_l_found = tick_time
                    if sl_l_found == 0 and tick_low <= sl_l:
                        sl_l_found = tick_time

                    if pt_l_found != 0 or sl_l_found != 0:
                        long_active = False  # どちらかのバリアに当たったら終了

            # --- ショート判定 ---
            if short_active:
                if tick_time > t1_s:
                    short_active = False  # TD超過
                else:
                    if pt_s_found == 0 and tick_low <= pt_s:  # ショートのPTは安値側
                        pt_s_found = tick_time
                    if sl_s_found == 0 and tick_high >= sl_s:  # ショートのSLは高値側
                        sl_s_found = tick_time

                    if pt_s_found != 0 or sl_s_found != 0:
                        short_active = False  # どちらかのバリアに当たったら終了

            # 早期退出: ロングもショートも決着がついていればループを抜ける
            # （durationのNumba内計算は削除し、breakのみを維持）
            if not long_active and not short_active:
                break

        out_pt_long[i] = pt_l_found
        out_sl_long[i] = sl_l_found
        out_pt_short[i] = pt_s_found
        out_sl_short[i] = sl_s_found

    # duration配列を削除し、純粋な到達時刻(マイクロ秒)の4つだけを返す
    return out_pt_long, out_sl_long, out_pt_short, out_sl_short


# --- ▲▲▲ Numba 双方向走査エンジンここまで ▲▲▲ ---

# =========================================================================
# ProxyLabelingEngine クラス前半 (初期化・データロード処理)
# =========================================================================


class ProxyLabelingEngine:
    """Engine to create a dual-labeled subset of data for proxy model training."""

    def __init__(self, config: ProxyLabelConfig):
        self.config = config
        logging.info(f"Using output directory: {self.config.output_dir}")

        # --- [修正] ロング・ショートそれぞれの集計用辞書を用意 ---
        self.label_counts_long: Dict[int, int] = {1: 0, 0: 0}
        self.label_counts_short: Dict[int, int] = {1: 0, 0: 0}

        self.report_data: List[Dict[str, Any]] = []
        self._validate_paths()
        self._validate_config()

    def _validate_paths(self):
        if not self.config.input_dir.exists():
            raise FileNotFoundError(
                f"Input directory not found: {self.config.input_dir}"
            )
        if not self.config.price_data_source_dir.exists():
            raise FileNotFoundError(
                f"Price data source dir not found: {self.config.price_data_source_dir}"
            )

    def _validate_config(self):
        cfg = self.config
        if cfg.filter_mode == "year" and cfg.filter_year is None:
            raise ValueError(
                "Filter mode 'year' requires a specific year (--year YYYY)."
            )
        if cfg.filter_mode == "month":
            if cfg.filter_year is None:
                raise ValueError(
                    "Filter mode 'month' requires a specific year (--year YYYY or via YYYY/MM)."
                )
            if cfg.filter_month is None or not (1 <= cfg.filter_month <= 12):
                raise ValueError(
                    "Filter mode 'month' requires a valid month number (1-12, via YYYY/MM)."
                )
        if cfg.filter_mode not in ["year", "month", "all"]:
            raise ValueError(
                f"Invalid filter_mode: {cfg.filter_mode}. Choose 'year', 'month', or 'all'."
            )

    def _get_duration_in_minutes(self, duration_str: str) -> float:
        """Converts a duration string (e.g., '300m', '90s') to a float value in minutes."""
        match = re.match(r"^(\d+)([ms])$", duration_str)
        if not match:
            logging.warning(
                f"Unexpected duration format: {duration_str}. Trying to parse as minutes."
            )
            num_match = re.match(r"(\d+)", duration_str)
            if num_match:
                return float(num_match.group(1))
            return 0.0
        value, unit = match.groups()
        value_float = float(value)
        if unit == "m":
            return value_float
        elif unit == "s":
            return value_float / 60.0
        return 0.0

    # =========================================================================
    # S5/S2読み込み・ファイル探索 (M1限定最適化)
    # =========================================================================

    def _discover_feature_paths(self) -> List[Path]:
        logging.info(
            f"Recursively searching for feature paths in {self.config.input_dir}..."
        )
        discovered_paths = list(self.config.input_dir.rglob("features_*_neutralized*"))
        feature_paths = [
            p for p in discovered_paths if p.is_dir() or p.name.endswith(".parquet")
        ]
        if not feature_paths:
            raise FileNotFoundError(
                f"No feature paths found in {self.config.input_dir}."
            )
        logging.info(f"  -> Found {len(feature_paths)} feature paths.")
        return feature_paths

    def _build_unified_lazyframe(
        self, feature_paths: List[Path]
    ) -> Tuple[pl.LazyFrame, pl.LazyFrame]:
        """
        S5の特徴量データを取得。要件に基づき M1 以外のタイムフレームはスキップしメモリを節約。
        """
        all_lazy_frames_hive = []
        all_lazy_frames_file = []

        timeframe_pattern = re.compile(
            r"features_e\d+[a-z]?_([a-zA-Z0-9\.]+)(?:_neutralized)?"
        )
        cfg = self.config

        logging.info(
            f"  -> Separating Hive/Files for S5 (Filtering strictly for {TARGET_TIMEFRAMES})..."
        )
        for path in feature_paths:
            name_to_match = path.stem if path.is_file() else path.name
            match = timeframe_pattern.search(name_to_match)
            if not match:
                continue
            timeframe = match.group(1)

            timeframe_suffix = f"_{timeframe}"
            lf: Optional[pl.LazyFrame] = None

            # Hive partition scan
            if path.is_dir():
                scan_base_path = path
                if cfg.filter_mode == "year" and cfg.filter_year is not None:
                    scan_base_path = scan_base_path / f"year={cfg.filter_year}"
                elif (
                    cfg.filter_mode == "month"
                    and cfg.filter_year is not None
                    and cfg.filter_month is not None
                ):
                    scan_base_path = (
                        scan_base_path
                        / f"year={cfg.filter_year}/month={cfg.filter_month}"
                    )

                if scan_base_path.exists():
                    try:
                        lf = pl.scan_parquet(str(scan_base_path / "**/*.parquet"))
                    except Exception as e:
                        logging.warning(f"Failed to scan {scan_base_path}: {e}")
                        continue

                if lf is not None:
                    lf_renamed = self._rename_features(lf, timeframe_suffix)
                    if lf_renamed is not None:
                        # [JOINASOF-FIX §11.34.16-O2] 旧 index シフトは撤去。
                        # 高TF/低TF の足選択は後段の close_ts ベース join_asof で
                        # 統一的に行う (index シフトは 1 対多=ffill を表現できず
                        # M5/M8/M15 を実質スパースにしていた)。ここでは縦持ち
                        # (timeframe 列付与) のまま渡す。
                        all_lazy_frames_hive.append(
                            lf_renamed.with_columns(
                                pl.lit(timeframe).alias("timeframe")
                            )
                        )

            # Single file scan
            elif path.is_file():
                try:
                    lf_full = pl.scan_parquet(str(path))
                    if "timestamp" not in lf_full.collect_schema().names():
                        continue
                except Exception as e:
                    logging.warning(f"Failed to scan file {path}: {e}")
                    continue

                date_filter: Optional[pl.Expr] = None
                if cfg.filter_mode == "year" and cfg.filter_year is not None:
                    date_filter = pl.col("timestamp").dt.year() == cfg.filter_year
                elif (
                    cfg.filter_mode == "month"
                    and cfg.filter_year is not None
                    and cfg.filter_month is not None
                ):
                    date_filter = (pl.col("timestamp").dt.year() == cfg.filter_year) & (
                        pl.col("timestamp").dt.month() == cfg.filter_month
                    )

                lf = lf_full.filter(date_filter) if date_filter is not None else lf_full

                if lf is not None:
                    lf_renamed = self._rename_features(lf, timeframe_suffix)
                    if lf_renamed is not None:
                        # [JOINASOF-FIX §11.34.16-O2] 旧 index シフトは撤去 (上記 hive 側と同旨)
                        all_lazy_frames_file.append(
                            lf_renamed.with_columns(
                                pl.lit(timeframe).alias("timeframe")
                            )
                        )

        unified_hive_lf = pl.LazyFrame()
        if all_lazy_frames_hive:
            unified_hive_lf = pl.concat(all_lazy_frames_hive, how="diagonal")
            logging.info(
                f"  -> Prepared S5 Hive LazyFrame ({len(all_lazy_frames_hive)} sources)."
            )
        else:
            logging.warning(f"No S5 Hive data found for {TARGET_TIMEFRAMES}.")

        unified_file_lf = pl.LazyFrame()
        if all_lazy_frames_file:
            unified_file_lf = pl.concat(all_lazy_frames_file, how="diagonal")
            logging.info(
                f"  -> Prepared S5 non-tick LazyFrame ({len(all_lazy_frames_file)} sources)."
            )
        else:
            logging.warning(f"No S5 non-tick data found for {TARGET_TIMEFRAMES}.")

        return unified_hive_lf, unified_file_lf

    # =========================================================================
    # [LOOKAHEAD-FIX §11.34.16] 高 TF 特徴量の「閉じたバー」再ラベル
    # -------------------------------------------------------------------------
    # 問題: 全 TF を label=left のまま diagonal concat → group_by(timestamp) で
    #   同ラベル融合すると、行 L (行動時刻 = L + ACTION_HORIZON_SEC) に
    #   「ラベル L のバー [L, L+tf)」の特徴量が乗る。tf > ACTION_HORIZON_SEC の
    #   TF (M5/M8/M15) ではバーのクローズ (L+tf) が行動時刻より未来になり、
    #   学習特徴量に最大 tf-180 秒の未来情報が混入していた (M5:+2分/M8:+5分/M15:+12分)。
    #   実証: S6 境界行の M15 wick 値 = 形成中バー [L, L+15分) の値と 17/17 bit 一致
    #   (2026-06-11 検証、レポート §11.34.16)。
    # 対処: tf > ACTION_HORIZON_SEC の TF のみ timestamp を +tf シフトし、
    #   ラベル L に「L で閉じたバー [L-tf, L)」を割り当てる。
    #   → 行動時刻 L+180 で完全に閉じた情報のみになる (因果的)。
    #   → 本番は最新の閉じたバーを常に持つため、初めて学習を 100% 再現可能になる
    #     (main.py HF-NB-GATE の minute_idx を T-180 基準に直す修正とセット)。
    # 注意:
    #   - M0.5/M1/M3 (tf <= 180s) はシフト禁止。これらはデータ到達点が行動時刻
    #     以下で既に因果的であり、シフトすると 1 本古い情報になり本番とズレる。
    #   - シフト後もグリッドは不変 (M15 ラベルは 900 の倍数のまま) なので、
    #     値が乗る行 (境界行) は変わらず、値の中身だけが閉じたバーになる。
    #   - 価格系 (entry/barrier 用 close/high/low) は S1_PROCESSED から別経路で
    #     取得しており (_load_all_price_data → price_window)、本シフトの影響なし。
    # =========================================================================
    ACTION_HORIZON_SEC = 180  # トリガー行の行動猶予 = TARGET_TIMEFRAMES の TF 秒数
    #   (TARGET_TIMEFRAMES を M3 以外に変える場合はここも合わせて変更すること)

    @staticmethod
    def _tf_to_seconds(timeframe: str) -> Optional[int]:
        """'M0.5'→30, 'M15'→900, 'H1'→3600, 'W1'→604800, 'MN'→約28日。未知形式は None。
        ※ [JOINASOF-FIX §11.34.16-O2] close_ts = ラベル + tf_sec の計算に使用。
           M0.5〜M15 は厳密値が必要 (MN は判定用近似だが LF はモデル不使用)。"""
        m = re.fullmatch(r"M(\d+(?:\.\d+)?)", timeframe)
        if m:
            return int(float(m.group(1)) * 60)
        m = re.fullmatch(r"H(\d+(?:\.\d+)?)", timeframe)
        if m:
            return int(float(m.group(1)) * 3600)
        m = re.fullmatch(r"W(\d+)", timeframe)
        if m:
            return int(m.group(1)) * 604800
        m = re.fullmatch(r"D(\d+)", timeframe)
        if m:
            return int(m.group(1)) * 86400  # D系: 念のため対応 (LF は通常 S5 不在)
        if timeframe == "MN":
            return 28 * 86400  # 月足: 可変長だが判定用の下限値で十分
        return None

    def _join_tf_to_triggers_asof(
        self,
        trigger_df: pl.DataFrame,
        tf_feat_df: pl.DataFrame,
        timeframe: str,
        feat_cols: List[str],
    ) -> pl.DataFrame:
        """[案 2 §11.34.16-Q] TF 足をトリガー行に割付。高 TF はスパース、低 TF は held。

        Q 節の BT 結果 (高 TF held=Y はスパース=X よりリスク構造劣位・AUC 同一) を受け、
        TF タイプで割付方式を分岐する:
          - 高 TF (tf > ACTION_HORIZON, M5/M8/M15): index シフト + exact join =
            「境界だけ実値・他 0」 のスパース表現 (本番 HF-NB-GATE と一致)。X を維持。
          - 低 TF (tf < ACTION_HORIZON, M0.5/M1): join_asof(backward, close_ts<=T) =
            「T で閉じた最新足」 を割付。本番 rfe L214 と一字一句一致し、足選択ズレ
            (M0.5=5 本/M1=2 本) を解消。低 TF は毎トリガーで値が変わるため held の
            連続エントリー慣性は生じない。
          - トリガー TF (tf == ACTION_HORIZON, M3): ラベル exact 一致 ([L,L+180) の
            close=L+180=T、本番と同一)。

        低 TF の join_asof は本番一致:
          - 各 TF 足のクローズ時刻 close_ts = ラベル + tf_sec
          - トリガーのエントリー時刻 T = ラベル(L_m3) + ACTION_HORIZON_SEC
          - join_asof(left=T, right=close_ts, backward) = 「close_ts <= T の最新足」
        単体検証済み (verified_joinasof_logic.py): M1 全行で期待ラベル一致、lookahead 違反 0。

        Args:
            trigger_df: [timestamp] (= M3 ラベル L_m3、本関数内でソート)
            tf_feat_df: [timestamp(=足ラベル)] + feat_cols (本関数内でソート)
            timeframe:  当該 TF 名
            feat_cols:  結合する特徴量列 (TF サフィックス付き、例 e1d_xxx_M15)
        Returns:
            trigger_df の各行に feat_cols を割付した DataFrame (高 TF=スパース/低 TF=held)。
        """
        tf_sec = self._tf_to_seconds(timeframe)
        if tf_sec is None:
            logging.warning(
                f"  [JOINASOF-FIX] 未知の TF '{timeframe}' — 割付スキップ (要確認)"
            )
            return trigger_df

        if tf_sec == self.ACTION_HORIZON_SEC:
            # トリガー TF (M3): ラベル exact 一致
            return trigger_df.join(
                tf_feat_df.select(["timestamp"] + feat_cols), on="timestamp", how="left"
            )

        if tf_sec > self.ACTION_HORIZON_SEC:
            # ════════════════════════════════════════════════════
            # [案 2 §11.34.16-Q] 高 TF (M5/M8/M15) は X (スパース) を維持。
            # Q 節 BT で held(Y) より X(スパース) がリスク構造優位 (MaxDD 4.80<6.79%、
            # 連敗 2<3) かつ AUC 同一と判明したため、高 TF は旧 index シフト方式に戻す。
            # ────────────────────────────────────────────────────
            # index シフト (shift(-1)) で「ラベル L の行 = L 時点で閉じた最新バー」 に
            # した上で、トリガー (M3 グリッド) に exact join。M3 グリッド (180s) と
            # 高 TF グリッド (300/480/900s) の交点だけ実値が乗り、非交点は null
            # (後段 fill_null(0) で 0)。これが本番 HF-NB-GATE (非境界 0 化) と一致する
            # スパース表現。held(join_asof backward) と違い「境界だけ実値」 になる。
            # ※ 本番側は main.py の HF-NB-GATE 復活とセット (学習スパース=本番スパース)。
            # ════════════════════════════════════════════════════
            shifted = (
                tf_feat_df.sort("timestamp")
                .with_columns(pl.col("timestamp").shift(-1).alias("timestamp"))
                .filter(pl.col("timestamp").is_not_null())
                .select(["timestamp"] + feat_cols)
            )
            return trigger_df.join(shifted, on="timestamp", how="left")

        # 低 TF (M0.5/M1, tf < 180): close_ts <= T の最新足を join_asof(backward)。
        # 本番 (rfe latest_features_cache: close_ts<=確定時刻の最新足) と足選択一致。
        # 低 TF は毎トリガーで値が変わる (M3 以下で必ず新足が確定) ため、held による
        # 連続エントリー慣性 (高 TF の Y で問題化) は生じず、純粋に足選択の正解化のみ。
        tf2 = (
            tf_feat_df.with_columns(
                (pl.col("timestamp") + pl.duration(seconds=tf_sec)).alias("close_ts")
            )
            .sort("close_ts")
            .select(["close_ts"] + feat_cols)
        )
        trig2 = trigger_df.with_columns(
            (pl.col("timestamp") + pl.duration(seconds=self.ACTION_HORIZON_SEC)).alias(
                "_T"
            )
        ).sort("_T")
        joined = trig2.join_asof(
            tf2, left_on="_T", right_on="close_ts", strategy="backward"
        )
        # [lookahead 監査] join_asof が引いた close_ts は必ず <= _T のはず (backward の定義)。
        # 念のため明示検査 (2_G の同時刻=未来バー混入の再発防止)。
        if "close_ts" in joined.columns:
            viol = joined.filter(
                pl.col("close_ts").is_not_null() & (pl.col("close_ts") > pl.col("_T"))
            ).height
            if viol > 0:
                raise RuntimeError(
                    f"[JOINASOF-FIX] LOOKAHEAD VIOLATION: TF={timeframe} で "
                    f"{viol} 行が close_ts > T。結合キー/方向を確認せよ。"
                )
        return joined.select(["timestamp"] + feat_cols).sort("timestamp")

    def _rename_features(
        self, lf: pl.LazyFrame, timeframe_suffix: str
    ) -> Optional[pl.LazyFrame]:
        try:
            current_schema = lf.collect_schema()
            if not current_schema.names():
                return None
            feature_cols = [col for col in current_schema.names() if col != "timestamp"]
            rename_exprs = [
                pl.col(col).alias(f"{col}{timeframe_suffix}") for col in feature_cols
            ]
            select_exprs = [pl.col("timestamp").cast(pl.Datetime("us", "UTC"))]
            if rename_exprs:
                select_exprs.extend(rename_exprs)
            return lf.select(select_exprs)
        except Exception as e:
            logging.warning(f"Failed to rename features: {e}")
            return None

    def _load_all_price_data(self) -> Dict[str, Any]:
        """S1_PROCESSEDからWilder平滑化でATR絶対値・ATR Ratioを自前計算して返す。
        tick価格データは月次チャンクループ内で直接スキャンするためここでは扱わない。"""
        tick_dir = S1_RAW_TICK_PARTITIONED
        if not tick_dir.exists():
            raise FileNotFoundError(f"Master tick directory not found: {tick_dir}")
        logging.info(f"  -> Confirmed tick source: '{tick_dir}' (will be loaded per-month chunk).")

        # --- S1_PROCESSEDのOHLCVからWilder平滑化でATR絶対値を自前計算 ---
        # e1c_atr_13はATR/ATR_13の相対値（≈1.0）のため使用不可
        # atr_ratioも全期間データで事前計算する（日次ループ内での計算は精度・速度ともに問題あり）
        all_atr_lfs = []
        for tf in TARGET_TIMEFRAMES:
            price_dir_tf = S1_PROCESSED / f"timeframe={tf}"
            if not price_dir_tf.exists():
                logging.warning(f"  -> S1_PROCESSED/timeframe={tf} が見つかりません。スキップします。")
                continue
            target_atr_name = f"e1c_atr_{ATR_PERIOD}_{tf}"
            atr_ratio_name = f"atr_ratio_{tf}"
            baseline_period = timeframe_bars_per_day.get(tf, 1440) * ATR_BASELINE_DAYS
            # 効率比の窓（実時間で60分・75-80分になる本数）と、その分表示用
            er_windows = ER_WINDOWS_BY_TF.get(tf, (20, 25))
            _tf_minutes = 1440.0 / timeframe_bars_per_day.get(tf, 1440)
            d_atr_name = f"d_atr_{tf}"

            # [DISC-FLAG 対応] s1_1_B が出力する disc 列を読み込み、
            #   不連続バー (disc=True) では前バーcloseを使わず H-L のみで TR を計算する。
            #   これにより週末跨ぎや祝日のギャップが ATR を異常値で汚染するのを防ぐ。
            #   本番側 core_indicators.calculate_barrier_atr と同じ思想 (Train-Serve Skew Free)。
            atr_lf = (
                pl.scan_parquet(str(price_dir_tf / "*.parquet"))
                # ★ d_atr = (close - open)/ATR のために open も読む
                .select(["timestamp", "open", "high", "low", "close", "disc"])
                .with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))
                .sort("timestamp")
                .with_columns([
                    # disc=True の足は H-L のみ、それ以外は通常の True Range
                    pl.when(pl.col("disc"))
                    .then(pl.col("high") - pl.col("low"))
                    .otherwise(
                        pl.max_horizontal(
                            pl.col("high") - pl.col("low"),
                            (pl.col("high") - pl.col("close").shift(1)).abs(),
                            (pl.col("low") - pl.col("close").shift(1)).abs(),
                        )
                    )
                    .ewm_mean(alpha=1 / ATR_PERIOD, adjust=False)
                    .alias(target_atr_name)
                ])
                # ATR Ratioも全期間データで計算（日次ループ内での計算より精度・速度ともに優れる）
                .with_columns([
                    (
                        pl.col(target_atr_name) /
                        (pl.col(target_atr_name).rolling_mean(window_size=baseline_period, min_samples=1) + 1e-10)
                    ).alias(atr_ratio_name)
                ])
                # --- 効率比（Efficiency Ratio）---
                #   ER(K) = |close[t] − close[t−K]| / Σ|close[i] − close[i−1]|
                #   すべて現在バーまでの情報なので、エントリー時刻に既知。未来を見ない。
                #   生の値のまま出す（閾値をモデルが学べるように）。
                .with_columns([
                    (pl.col("close") - pl.col("close").shift(1)).abs().alias("_er_absmove")
                ])
                .with_columns([
                    (
                        (pl.col("close") - pl.col("close").shift(k)).abs()
                        / (
                            pl.col("_er_absmove").rolling_sum(window_size=k, min_samples=k)
                            + 1e-12
                        )
                    ).alias(f"eff_ratio_{k}_{tf}")
                    for k in er_windows
                ])
                # --- 符号つき1バー変位 d/ATR ---
                #   M3バー [L, L+180) では open=P(L)、close=P(L+180) なので
                #   これは測定で使った d = P(L+180) − P(L) を ATR で割ったものと同一。
                #   既存の e1d_body_size_atr は abs() 付きで符号が落ちているため別途必要。
                .with_columns([
                    (
                        (pl.col("close") - pl.col("open"))
                        / (pl.col(target_atr_name) + 1e-10)
                    ).alias(d_atr_name)
                ])
                .select(
                    ["timestamp", target_atr_name, atr_ratio_name, d_atr_name]
                    + [f"eff_ratio_{k}_{tf}" for k in er_windows]
                )
            )
            all_atr_lfs.append(atr_lf)
            logging.info(
                f"  -> Prepared ATR blueprint: S1_PROCESSED/timeframe={tf} -> '{target_atr_name}' + '{atr_ratio_name}' (baseline={baseline_period}bars, disc-aware)"
            )
            logging.info(
                f"  -> Prepared Efficiency Ratio: {[f'eff_ratio_{k}_{tf}' for k in er_windows]} "
                f"(窓 {er_windows}本 = {[int(k * _tf_minutes) for k in er_windows]}分)"
            )
            logging.info(
                f"  -> Prepared signed bar displacement: '{d_atr_name}' = (close - open) / ATR"
            )

        if not all_atr_lfs:
            raise ValueError(
                f"FATAL: No valid ATR columns could be computed for {TARGET_TIMEFRAMES} from S1_PROCESSED."
            )

        return {"atr_lfs": all_atr_lfs}

    # =========================================================================
    # ヘルパー関数群 (パーティション探索・時間計算)
    # =========================================================================

    def _discover_partitions(self, unified_lf: pl.LazyFrame) -> pl.DataFrame:
        if unified_lf.collect_schema().names() == []:
            return pl.DataFrame({"date": []}).select(pl.col("date").cast(pl.Date))
        df_dates = (
            unified_lf.select(pl.col("timestamp").dt.date().alias("date"))
            .unique()
            .collect()
        )
        if df_dates.is_empty():
            return df_dates.select(pl.col("date").cast(pl.Date))
        return df_dates.sort("date")

    def _get_bar_duration_minutes(self, timeframe: str) -> float:
        if timeframe == "tick":
            return 0.0
        if timeframe == "MN":
            return 1.0 * 43200
        value_match = re.search(r"(\d*\.?\d+)", timeframe)
        unit_match = re.search(r"([A-Z])", timeframe)
        if not value_match or not unit_match:
            return 0.0
        try:
            value = float(value_match.group(1)) if value_match.group(1) else 1.0
        except ValueError:
            return 0.0
        unit = unit_match.group(1)
        if unit == "M":
            return value
        if unit == "H":
            return value * 60
        if unit == "D":
            return value * 1440
        if unit == "W":
            return value * 10080
        return 0.0

    # =========================================================================
    # メイン実行ループ (run)
    # =========================================================================

    def run(self):
        logging.info(f"### Phase 3: Final Labeling (V5 Dual-Directional Labeling) ###")
        logging.info(f"Applying filter: {self.config.get_filter_description()}")
        logging.info(f"Target Timeframes: {TARGET_TIMEFRAMES}")
        logging.info(
            f"ATR Ratio Filter: atr_ratio >= {ATR_RATIO_THRESHOLD} (Period: {ATR_PERIOD}, Baseline: {ATR_BASELINE_DAYS} day)"
        )
        logging.info(
            f"Long Rule : PT={RULE_LONG['pt_mult']}, SL={RULE_LONG['sl_mult']}, TD={RULE_LONG['td']}"
        )
        logging.info(
            f"Short Rule: PT={RULE_SHORT['pt_mult']}, SL={RULE_SHORT['sl_mult']}, TD={RULE_SHORT['td']}"
        )

        cfg = self.config

        if not cfg.resume and cfg.output_dir.exists():
            logging.info(
                f"Resume is disabled. Deleting existing directory: {cfg.output_dir}"
            )
            shutil.rmtree(cfg.output_dir)
        cfg.output_dir.mkdir(parents=True, exist_ok=True)

        try:
            logging.info("Step 1: Discovering feature paths...")
            all_feature_paths = self._discover_feature_paths()

            logging.info("Step 2: Building unified bet/feature frames (Lazy)...")
            unified_hive_lf, unified_file_lf = self._build_unified_lazyframe(
                all_feature_paths
            )

            if (
                unified_hive_lf.collect_schema().names() == []
                and unified_file_lf.collect_schema().names() == []
            ):
                logging.warning(
                    f"No data found for filter '{cfg.get_filter_description()}'. Exiting."
                )
                self._generate_report()
                return

            logging.info("Step 3: Preparing price data blueprints...")
            price_components = self._load_all_price_data()
            atr_lfs = price_components["atr_lfs"]

            # max_lookahead_minutesを先に定義（月チャンクのマージン計算に使用）
            max_lookahead_minutes = max(
                self._get_duration_in_minutes(RULE_LONG["td"]),
                self._get_duration_in_minutes(RULE_SHORT["td"]),
            )
            max_lookahead_delta = dt.timedelta(minutes=max_lookahead_minutes)
            # 月末に加算するルックアヘッドマージン（TD分 + 安全バッファ3日）
            lookahead_margin = dt.timedelta(minutes=max_lookahead_minutes) + dt.timedelta(days=3)

            # ATRは全期間・全時間足で1回だけ事前ロード（軽量なのでOK）
            logging.info(f"   -> Pre-loading {len(atr_lfs)} ATR files into memory...")
            atr_dfs = [lf.collect().sort("timestamp") for lf in atr_lfs]

            logging.info("Step 4: Discovering daily partitions for processing...")
            partitions_df_hive = self._discover_partitions(unified_hive_lf)
            partitions_df_file = self._discover_partitions(unified_file_lf)
            partitions_df = (
                pl.concat([partitions_df_hive, partitions_df_file])
                .unique()
                .sort("date")
            )

            logging.info(f"   -> Found {len(partitions_df)} daily partitions.")
            if partitions_df.is_empty():
                logging.warning("No partitions found. Exiting.")
                self._generate_report()
                return

            # 処理対象の年月一覧を作成（外側ループ用）
            months_df = (
                partitions_df.select(
                    pl.col("date").dt.year().alias("year"),
                    pl.col("date").dt.month().alias("month"),
                )
                .unique()
                .sort(["year", "month"])
            )

            logging.info(
                f"Step 5: Starting monthly chunked processing loop... "
                f"({len(months_df)} months / {len(partitions_df)} days)"
            )

            # =========================================================
            # 外側ループ：月単位でtickをロード→処理→破棄
            # =========================================================
            for month_row in tqdm(
                months_df.iter_rows(named=True),
                total=len(months_df),
                desc="Processing Months",
            ):
                y, m = month_row["year"], month_row["month"]
                _, last_day = calendar.monthrange(y, m)

                # その月のtick範囲（ルックアヘッドマージン付き）
                month_start = dt.datetime(y, m, 1, tzinfo=dt.timezone.utc)
                month_end = (
                    dt.datetime(y, m, last_day, tzinfo=dt.timezone.utc)
                    + lookahead_margin
                )

                # ★ その月のtickデータだけをメモリに載せる
                # hive_partitioning=Trueで述語プッシュダウンを確実に有効化
                # 用途: (a) バリア hit 判定の tick 走査 (high/low、価格 = mid)、
                #       (b) バリア基準 close のアンカー = エントリー時刻 L+180 の mid
                #           (ENTRY-ANCHOR FIX §11.34.16-O3、_calculate_labels_for_batch 内で
                #            timestamp+ACTION_HORIZON_SEC を backward asof)。
                # mid_price を close/high/low に展開 (tick は単一点なので OHLC 同値)。
                logging.debug(f"Loading tick chunk for {y}-{m:02d}...")
                try:
                    # [PERF-FIX 2026-07-08] hive パーティション列 (year/month) で
                    # 先に枝刈りしてから timestamp フィルタする。
                    # 旧実装は timestamp (非パーティション列) だけで filter しており、
                    # 述語プッシュダウンが効かず全 hive ファイルを開いてから行 filter
                    # していた可能性が高い。パーティション列で当月 (+ lookahead が翌月頭に
                    # 食い込むぶん翌月) に限定すると、開く Parquet ファイル自体が減り
                    # I/O・メモリとも軽くなる。timestamp フィルタは境界の端数を落とす
                    # ために薄く残す (パーティション枝刈りの後段なのでコストは僅少)。
                    #   月末 + lookahead_margin が翌月に食い込むケースを被覆するため、
                    #   当月と翌月の (year, month) を許可集合に含める。
                    _next_m = m + 1 if m < 12 else 1
                    _next_y = y if m < 12 else y + 1
                    _allowed_ym = [(y, m), (_next_y, _next_m)]
                    base_price_chunk_df = (
                        pl.scan_parquet(
                            str(S1_RAW_TICK_PARTITIONED / "**/*.parquet"),
                            hive_partitioning=True,
                        )
                        .filter(
                            pl.struct(["year", "month"]).is_in(
                                [{"year": _yy, "month": _mm} for _yy, _mm in _allowed_ym]
                            )
                        )
                        .rename({"datetime": "timestamp"})
                        .select("timestamp", "mid_price")
                        .with_columns(
                            pl.col("timestamp").cast(pl.Datetime("us", "UTC")),
                            pl.col("mid_price").alias("close"),
                            pl.col("mid_price").alias("high"),
                            pl.col("mid_price").alias("low"),
                        )
                        .select("timestamp", "close", "high", "low")
                        .filter(pl.col("timestamp").is_between(month_start, month_end))
                        .collect()
                        .unique("timestamp", keep="first")
                        .sort("timestamp")
                    )
                except Exception as e:
                    logging.warning(f"Failed to load tick chunk for {y}-{m:02d}: {e}. Skipping month.")
                    continue

                if base_price_chunk_df.is_empty():
                    logging.warning(f"No tick data found for {y}-{m:02d}. Skipping month.")
                    continue

                logging.debug(
                    f"  -> Tick chunk {y}-{m:02d}: {len(base_price_chunk_df):,} rows loaded."
                )

                # その月に含まれる日のリストを取得
                days_in_month = partitions_df.filter(
                    (pl.col("date").dt.year() == y) &
                    (pl.col("date").dt.month() == m)
                )

                # =====================================================
                # [JOINASOF-FIX §11.34.16-O2] 月次の縦持ち TF フレームを準備。
                # join_asof 割付は日初トリガーが前日夜の高TF足を引く必要があるため、
                # TF 特徴量は月次チャンク全体を使う (トリガー行は日次)。
                # ※ unified_*_lf は全期間 LazyFrame のため、月初の held 窓〜月末に
                #   絞ってから collect する (全期間フルマテリアライズの OOM/低速を回避)。
                #   月初トリガーが前月末の足を引くのに必要な遡り = 最大 held 窓。
                #   モデル使用 TF は M15 (900s) までだが、年末年始等の連休が月境界に
                #   重なる極端ケースまで安全に潰すため 4 日遡る (コストはほぼゼロ)。
                # =====================================================
                _held_lookback = dt.timedelta(days=4)
                _win_start = month_start - _held_lookback
                _monthly_hive_df = (
                    unified_hive_lf.filter(
                        pl.col("timestamp").is_between(_win_start, month_end)
                    ).collect()
                    if "timestamp" in unified_hive_lf.collect_schema().names()
                    else pl.DataFrame()
                )
                _monthly_file_df = (
                    unified_file_lf.filter(
                        pl.col("timestamp").is_between(_win_start, month_end)
                    ).collect()
                    if "timestamp" in unified_file_lf.collect_schema().names()
                    else pl.DataFrame()
                )
                _monthly_bets_df = pl.concat(
                    [_monthly_hive_df, _monthly_file_df], how="diagonal"
                )
                # TF 別の特徴量フレームを事前に分離 (timeframe 列で振り分け、月次・ソート済)。
                # 各 TF: [timestamp(=足ラベル)] + その TF の特徴量列 (他 TF 由来の全 null 列は drop)。
                # ※ [複製バグ修正 §11.34.16-O2] S5 は 6 エンジン×6TF=36 ファイルを縦積みのため、
                #   同一 TF 内に同一 timestamp が複数エンジン分 (最大 6 行) 存在する。
                #   各行は自エンジンの列だけ実値・他 null。これを group_by(timestamp).first()
                #   で 1 行に集約しないと、後段の join_asof が 1 トリガーを 6 倍 (6TF で 36 倍)
                #   に複製する。従来の group_by("timestamp").agg(first()) が担っていた
                #   「エンジン横断の集約」 をここで TF 別に復元する (TF 融合は join_asof が担当)。
                _monthly_tf_frames: Dict[str, pl.DataFrame] = {}
                if not _monthly_bets_df.is_empty():
                    for _tf_name in _monthly_bets_df["timeframe"].unique().to_list():
                        _sub = (
                            _monthly_bets_df.filter(pl.col("timeframe") == _tf_name)
                            .drop("timeframe")
                            .sort("timestamp")
                        )
                        # 全 null 列 (他 TF 由来) を除外して当該 TF の実特徴量列のみ残す
                        _nonnull_cols = [
                            c
                            for c in _sub.columns
                            if c == "timestamp" or _sub[c].null_count() < _sub.height
                        ]
                        _sub = _sub.select(_nonnull_cols)
                        # [複製バグ修正] 同一 timestamp の複数エンジン行を 1 行に集約
                        # (各列の non-null 先頭値を拾う)。これで timestamp が一意になり、
                        # join_asof が 1 トリガー = 1 行を保つ。
                        _monthly_tf_frames[_tf_name] = (
                            _sub.group_by("timestamp")
                            .agg(pl.all().drop_nulls().first())
                            .sort("timestamp")
                        )

                # =====================================================
                # [PERF-FIX 2026-07-08] 日次トリガーを月次で 1 回だけ辞書化する。
                #
                # 旧実装は日次ループ内で unified_hive_lf / unified_file_lf
                # (全期間 = 約5年 の LazyFrame) に対し
                #   .filter(timestamp.dt.date() == current_date).collect()
                # を 4 回 (hive/file × 2) 呼んでいた。営業日 ~1,200 日ぶん
                # 「5 年分スキャン → 1 日抽出」を繰り返し = O(days × 全期間scan)
                # が 9h 級の支配項だった (Numba 走査でも join_asof でもない)。
                #
                # 対処: 既に月次で materialize 済みの _monthly_hive_df /
                # _monthly_file_df を日付キーで partition_by し、日次ループは
                # 辞書 get() で O(1) 参照する。
                #   - メモリ: ピークは「当月ぶんのトリガー行」= 既存 _monthly_*_df
                #     と同一実体を再利用するだけ。日次 collect の一時バッファが
                #     消えるぶ ん むしろ軽い (メモリリークしない)。
                #   - 境界: _monthly_*_df は _win_start(=月初-4日)〜month_end を含み、
                #     当月内の全日を完全被覆する (前月末 held 窓ぶんは余分に持つが害なし)。
                #   - 正当性: 日次 raw_bets_df は unified_*_lf 由来で、_monthly_*_df も
                #     同一 unified_*_lf を is_between で切り出したもの。当月日付では
                #     両者は行集合が一致する。
                # =====================================================
                _bets_by_day: Dict[dt.date, pl.DataFrame] = {}
                _monthly_bets_for_days = [
                    _df for _df in (_monthly_hive_df, _monthly_file_df)
                    if (not _df.is_empty()) and ("timestamp" in _df.columns)
                ]
                if _monthly_bets_for_days:
                    _monthly_bets_all = pl.concat(
                        _monthly_bets_for_days, how="diagonal"
                    ).with_columns(pl.col("timestamp").dt.date().alias("_d"))
                    for _dkey, _dframe in _monthly_bets_all.partition_by(
                        "_d", as_dict=True, include_key=False
                    ).items():
                        # partition_by の dict キーは tuple (単一キーでも)。正規化。
                        _dk = _dkey[0] if isinstance(_dkey, tuple) else _dkey
                        _bets_by_day[_dk] = _dframe
                    del _monthly_bets_all

                # =====================================================
                # 内側ループ：日次処理（既存ロジックをそのまま維持）
                # =====================================================
                for row in days_in_month.iter_rows(named=True):
                    current_date = row["date"]
                    year_d, month_d, day_d = (
                        current_date.year,
                        current_date.month,
                        current_date.day,
                    )

                    output_partition_dir = (
                        cfg.output_dir / f"year={year_d}/month={month_d}/day={day_d}"
                    )
                    if cfg.resume and output_partition_dir.exists():
                        logging.debug(
                            f"Resuming: Output exists for {current_date}. Skipping."
                        )
                        continue

                    # [PERF-FIX 2026-07-08] 全期間 LazyFrame への日次 collect を廃止し、
                    # 月次で 1 回だけ構築した日付辞書 _bets_by_day から O(1) 取得する。
                    # (旧: unified_*_lf.filter(date==current_date).collect() ×4)
                    raw_bets_df = _bets_by_day.get(current_date, pl.DataFrame())

                    if raw_bets_df.is_empty():
                        continue

                    # 1. 各タイムスタンプで有効なターゲット時間足を抽出（重複排除込み）
                    valid_targets_df = (
                        raw_bets_df.filter(pl.col("timeframe").is_in(TARGET_TIMEFRAMES))
                        .select(["timestamp", "timeframe"])
                        .unique()
                    )

                    # 2-3. [JOINASOF-FIX §11.34.16-O2] 各 TF を close_ts ベース join_asof で
                    # トリガー行に割付 (group_by 同一timestamp融合を置換)。
                    # 本番 (rfe latest_features_cache: close_ts<=確定時刻の最新足を ffill)
                    # と一字一句一致。トリガー行 = 当日の M3 ラベル、TF 特徴量 = 月次フレーム。
                    _trigger_rows = (
                        valid_targets_df.select("timestamp").unique().sort("timestamp")
                    )
                    daily_bets_df = valid_targets_df.sort(["timeframe", "timestamp"])
                    for _tf_name, _tf_frame in _monthly_tf_frames.items():
                        _feat_cols = [c for c in _tf_frame.columns if c != "timestamp"]
                        if not _feat_cols:
                            continue
                        _assigned = self._join_tf_to_triggers_asof(
                            _trigger_rows, _tf_frame, _tf_name, _feat_cols
                        )
                        # トリガー行 (timestamp) に当該 TF 特徴量を結合 (全ターゲット行へ)
                        daily_bets_df = daily_bets_df.join(
                            _assigned, on="timestamp", how="left"
                        )
                    daily_bets_df = daily_bets_df.sort(["timeframe", "timestamp"])

                    if daily_bets_df.is_empty():
                        continue

                    min_ts_req = daily_bets_df["timestamp"].min()
                    if min_ts_req is None:
                        continue

                    # ロング/ショートの最長TD + 余裕分(2日)で窓を切り出す
                    max_ts_req = min_ts_req + max_lookahead_delta + dt.timedelta(days=2)

                    # ★ base_price_chunk_df（月チャンク）から窓を切り出す
                    price_window_df = (
                        base_price_chunk_df.filter(
                            pl.col("timestamp").is_between(min_ts_req, max_ts_req)
                        )
                        .sort("timestamp")
                    )

                    if price_window_df.is_empty():
                        continue

                    for atr_df in atr_dfs:
                        atr_df_small = atr_df.filter(
                            pl.col("timestamp").is_between(min_ts_req, max_ts_req)
                        )
                        if not atr_df_small.is_empty():
                            price_window_df = price_window_df.join_asof(
                                atr_df_small, on="timestamp"
                            )

                    price_window_df = price_window_df.fill_null(strategy="forward")

                    daily_labeled_df = self._calculate_labels_for_batch(
                        daily_bets_df, price_window_df
                    )

                    if daily_labeled_df is not None and not daily_labeled_df.is_empty():
                        self._update_label_counts_dual(daily_labeled_df)
                        self._collect_report_data_dual(daily_labeled_df, current_date)
                        output_partition_dir.mkdir(parents=True, exist_ok=True)
                        daily_labeled_df.write_parquet(
                            output_partition_dir / "data.parquet", compression="zstd"
                        )

                    # 日次メモリ解放
                    del daily_bets_df, price_window_df, daily_labeled_df
                    # [BUGFIX] daily_bets_hive_df / daily_bets_file_df は過去の
                    #   リファクタで daily_bets_df に統合済み。定義が存在しないため
                    #   この行に到達すると NameError になる（Ruff F821）。削除する。
                    gc.collect()

                # ★ 月チャンク終了: tickデータをメモリから破棄
                del base_price_chunk_df
                gc.collect()
                logging.debug(f"  -> Tick chunk {y}-{m:02d} released from memory.")

            self._log_final_summary()
            self._generate_report()

        except Exception as e:
            logging.error(f"A critical error occurred: {e}", exc_info=True)
            raise

    # =========================================================================
    # コアロジック: ATR収縮フィルター適用と双方向ラベルの一括計算
    # =========================================================================

    def _calculate_labels_for_batch(
        self, daily_bets_df: pl.DataFrame, price_window_df: pl.DataFrame
    ) -> pl.DataFrame | None:
        if daily_bets_df.is_empty():
            return None

        if not NUMBA_AVAILABLE:
            logging.error("Numba is required for labeling but not found. Skipping.")
            return None

        try:
            ticks_df_np = price_window_df.select(
                pl.col("timestamp").cast(pl.Int64).alias("ticks_ts"),
                pl.col("high").alias("ticks_high"),
                pl.col("low").alias("ticks_low"),
            )
            ticks_ts_np = ticks_df_np["ticks_ts"].to_numpy()
            ticks_high_np = ticks_df_np["ticks_high"].to_numpy()
            ticks_low_np = ticks_df_np["ticks_low"].to_numpy()
        except Exception as e:
            logging.error(f"Failed to convert tick data to Numpy arrays: {e}")
            return None

        labeled_chunks = []
        for timeframe_tuple, group_df in daily_bets_df.group_by("timeframe"):
            timeframe = timeframe_tuple[0]
            if timeframe is None or group_df.is_empty():
                continue

            # 対象外のデータが混入した場合は安全のためスキップ
            if timeframe not in TARGET_TIMEFRAMES:
                logging.debug(
                    f"Skipping timeframe {timeframe} (Targets are {TARGET_TIMEFRAMES})."
                )
                continue

            # ロング/ショートそれぞれのタイムアウト（TD）を計算
            td_long_minutes = self._get_duration_in_minutes(RULE_LONG["td"])
            td_short_minutes = self._get_duration_in_minutes(RULE_SHORT["td"])

            # [LOOKAHEAD-FIX §11.34.16] TD の基点をラベル(timestamp=L)から
            # エントリー時刻 (L + ACTION_HORIZON_SEC) に変更。
            # 旧実装は L+TD で打ち切っており、TD30分 が実質「エントリー後27分」
            # になっていた (BT/本番はエントリー後 TD 分で判定するため不整合)。
            t1_max_long_expr = (
                pl.col("timestamp")
                + pl.duration(seconds=self.ACTION_HORIZON_SEC)
                + pl.duration(minutes=td_long_minutes)
            )
            t1_max_short_expr = (
                pl.col("timestamp")
                + pl.duration(seconds=self.ACTION_HORIZON_SEC)
                + pl.duration(minutes=td_short_minutes)
            )

            atr_col_name = f"e1c_atr_{ATR_PERIOD}_{timeframe}"
            atr_ratio_col_name = f"atr_ratio_{timeframe}"  # 事前計算済みカラム名
            if atr_col_name not in price_window_df.columns:
                logging.warning(
                    f"Required ATR column '{atr_col_name}' not found. Skipping."
                )
                continue
            if atr_ratio_col_name not in price_window_df.columns:
                logging.warning(
                    f"Required ATR ratio column '{atr_ratio_col_name}' not found. Skipping."
                )
                continue

            original_cols = [
                c for c in group_df.columns if c not in ["timestamp", "close"]
            ]

            # ===================================================================
            # [ENTRY-ANCHOR FIX §11.34.16-O3] バリア基準価格 = エントリー時刻の価格
            # ===================================================================
            # 旧実装は close/high/low/ATR を一括で timestamp=L (トリガー = M3 バー始値
            # 時刻) に join_asof(backward) し、バリア基準 close に「L 時点 (バー始値) の
            # tick mid」を採用していた。しかし本番のエントリーは
            #   T = L + ACTION_HORIZON_SEC (= L+180、M3 バーのクローズ時刻)
            # で行われ、エントリー価格 = realtime_feature_engine の
            #   current_price = data["close"][-1] = M3 バー [L, L+180) の close。
            # よって旧実装の基準 close は実エントリーより 1 バー (約 3 分) 手前で、
            # かつシグナルは「価格が動いている」ときに出るため必ずトレード方向に
            # 有利側へズレていた。走査は L+180 から始まる (entry_offset) ため、
            # [L, L+180) で既に起きた順行分が「タダ乗り」して PT に早期到達し、
            # ラベルが楽観化 → BT と本番の乖離 (本番 TO 過多) の震源になっていた。
            # 実測 (本番 ReportHistory vs BT detailed_trade_log、同一足 93 件) で
            # 本番 TO×BT-PT 24 件・基準価格が方向有利側に中央 +$6.1・24/24 方向一致を
            # 確認、コード (本ブロック + tick mid ソース) でも裏取り済み。
            #
            # 修正方針:
            #   - バリア基準 close = エントリー時刻 L+180 の価格に合わせる
            #     (price_window の tick mid を L+180 で backward asof = L+180 以前で
            #      最新の tick = M3 バー [L, L+180) の close = 本番エントリー価格)。
            #   - ATR / ATR_ratio は従来通り トリガー時刻 L で結合 (バー [L, L+180) の
            #     値で、本番ゲート整合済み)。変更しない。
            #   - 走査用 ticks_high/low (price_window 由来) と走査開始
            #     t0+ACTION_HORIZON_SEC、縦バリア t1_max = L+180+TD は既に正しく不変。
            #   - SPREAD 項は §2 で確定済みの意図的な保守化のため不変。
            _entry_offset = pl.duration(seconds=self.ACTION_HORIZON_SEC)

            # (1) ATR / ATR_ratio は トリガー時刻 L で結合 (バーの値、本番整合済み)
            #     ★追加特徴量（効率比・符号つきd）も同じ経路で持ち込む。
            #       いずれもバー [L, L+180) までの情報だけで作られており、
            #       エントリー時刻 L+180 には確定している。未来を見ていない。
            _extra_feature_cols = [
                c
                for c in price_window_df.columns
                if c.startswith("eff_ratio_") or c.startswith("d_atr_")
            ]
            bets_with_atr_df = group_df.join_asof(
                price_window_df.select(
                    ["timestamp", atr_col_name, atr_ratio_col_name] + _extra_feature_cols
                ),
                on="timestamp",
            ).filter(pl.col(atr_col_name).is_not_null())

            if bets_with_atr_df.is_empty():
                continue

            # 旧 close 列が混入していれば除去 (アンカーで付け直すため)
            if "close" in bets_with_atr_df.columns:
                bets_with_atr_df = bets_with_atr_df.drop("close")

            # (2) バリア基準 close は エントリー時刻 L+180 の価格を別途 asof 結合
            #     (本番 current_price = data["close"][-1] = M3 バー [L,L+180) close と一致)
            _anchor_price_df = price_window_df.select(
                pl.col("timestamp").alias("_anchor_ts"),
                pl.col("close"),
            ).sort("_anchor_ts")

            bets_with_price_df = (
                bets_with_atr_df.with_columns(
                    (pl.col("timestamp") + _entry_offset).alias("_entry_ts")
                )
                .sort("_entry_ts")
                .join_asof(
                    _anchor_price_df,
                    left_on="_entry_ts",
                    right_on="_anchor_ts",
                    strategy="backward",
                )
                .sort("timestamp")
            )
            # 一時列を除去 (バージョン差で右キーが残る場合に備え存在時のみ drop)
            _tmp_cols = [
                c for c in ["_entry_ts", "_anchor_ts"]
                if c in bets_with_price_df.columns
            ]
            if _tmp_cols:
                bets_with_price_df = bets_with_price_df.drop(_tmp_cols)

            # エントリー時刻の価格が取得できなかった行 (窓端等) は除外
            bets_with_price_df = bets_with_price_df.filter(
                pl.col("close").is_not_null()
            )

            if bets_with_price_df.is_empty():
                continue

            # atr_value・atr_ratioともに事前計算済みカラムをそのまま使用（日次ループ内での再計算なし）
            bets_df_with_atr = bets_with_price_df.with_columns(
                pl.col(atr_col_name).alias("atr_value"),
                pl.col(atr_ratio_col_name).alias("atr_ratio"),
            )

            # --- 全行を保持しつつ、ATR Ratioでエントリー起点に is_trigger=1 を付与 ---
            bets_df_all = bets_df_with_atr.with_columns(
                pl.when(pl.col("atr_ratio") >= ATR_RATIO_THRESHOLD)
                .then(1)
                .otherwise(0)
                .cast(pl.Int8)
                .alias("is_trigger")
            )

            # Numbaに渡して重い計算をするのは is_trigger==1 の行だけにする
            bets_df_filtered = bets_df_all.filter(pl.col("is_trigger") == 1)
            if bets_df_filtered.is_empty():
                continue

            # バリアの一括計算 (ロングとショート両方)
            bets_df = bets_df_filtered.select(
                pl.col("timestamp").alias("t0"),
                # ★修正: ロング用バリア（Askエントリー想定: PTは遠く、SLは近く）
                (
                    pl.col("close")
                    + pl.col("atr_value") * RULE_LONG["pt_mult"]
                    + SPREAD
                ).alias("pt_long"),
                (
                    pl.col("close")
                    - pl.col("atr_value") * RULE_LONG["sl_mult"]
                    + SPREAD
                ).alias("sl_long"),
                t1_max_long_expr.alias("t1_max_long"),
                # ★修正: ショート用バリア（Bidエントリー/Ask決済想定: PTは遠く、SLは近く）
                (
                    pl.col("close")
                    - pl.col("atr_value") * RULE_SHORT["pt_mult"]
                    - SPREAD
                ).alias("pt_short"),
                (
                    pl.col("close")
                    + pl.col("atr_value") * RULE_SHORT["sl_mult"]
                    - SPREAD
                ).alias("sl_short"),
                t1_max_short_expr.alias("t1_max_short"),
                pl.col("atr_value"),
                pl.col("atr_ratio"),  # ★ S6出力に含める（バックテストシミュレーターが再計算不要になる）
                pl.col("close"),
                pl.col(original_cols),
            )

            if bets_df.is_empty():
                continue

            try:
                bets_t0_np = bets_df["t0"].cast(pl.Int64).to_numpy()
                bets_t1_max_l_np = bets_df["t1_max_long"].cast(pl.Int64).to_numpy()
                bets_t1_max_s_np = bets_df["t1_max_short"].cast(pl.Int64).to_numpy()

                bets_pt_l_np = bets_df["pt_long"].to_numpy(writable=True)
                bets_sl_l_np = bets_df["sl_long"].to_numpy(writable=True)
                bets_pt_s_np = bets_df["pt_short"].to_numpy(writable=True)
                bets_sl_s_np = bets_df["sl_short"].to_numpy(writable=True)
            except Exception as e:
                logging.error(f"Failed to convert bets data to Numpy arrays: {e}")
                continue

            # Numbaによる双方向同時判定
            out_pt_l, out_sl_l, out_pt_s, out_sl_s = _numba_find_hits_dual(
                bets_t0_np,
                bets_t1_max_l_np,
                bets_t1_max_s_np,
                bets_pt_l_np,
                bets_sl_l_np,
                bets_pt_s_np,
                bets_sl_s_np,
                ticks_ts_np,
                ticks_high_np,
                ticks_low_np,
                np.int64(self.ACTION_HORIZON_SEC * 1_000_000),  # [LOOKAHEAD-FIX] エントリーオフセット(us)
            )

            # 計算済みトリガー行の DataFrame 作成
            calculated_df = (
                bets_df.with_columns(
                    pl.Series("pt_l_time", out_pt_l),
                    pl.Series("sl_l_time", out_sl_l),
                    pl.Series("pt_s_time", out_pt_s),
                    pl.Series("sl_s_time", out_sl_s),
                )
                .with_columns(
                    # ロング用ラベルと「正確な決済時刻(マイクロ秒)」の特定
                    label_long=pl.when(
                        (pl.col("pt_l_time") > 0)
                        & (
                            (pl.col("sl_l_time") == 0)
                            | (pl.col("pt_l_time") < pl.col("sl_l_time"))  # 🟢 修正
                        )
                    )
                    .then(pl.lit(1, dtype=pl.Int8))
                    .otherwise(pl.lit(0, dtype=pl.Int8)),
                    end_l=pl.when(
                        (pl.col("pt_l_time") > 0)
                        & (
                            (pl.col("sl_l_time") == 0)
                            | (pl.col("pt_l_time") < pl.col("sl_l_time"))  # 🟢 修正
                        )
                    )
                    .then(pl.col("pt_l_time"))
                    .when(pl.col("sl_l_time") > 0)
                    .then(pl.col("sl_l_time"))
                    .otherwise(
                        pl.col("t1_max_long").cast(pl.Int64)
                    ),  # ←変更: Int64にキャスト
                    # ショート用ラベルと「正確な決済時刻(マイクロ秒)」の特定
                    label_short=pl.when(
                        (pl.col("pt_s_time") > 0)
                        & (
                            (pl.col("sl_s_time") == 0)
                            | (pl.col("pt_s_time") < pl.col("sl_s_time"))  # 🟢 修正
                        )
                    )
                    .then(pl.lit(1, dtype=pl.Int8))
                    .otherwise(pl.lit(0, dtype=pl.Int8)),
                    end_s=pl.when(
                        (pl.col("pt_s_time") > 0)
                        & (
                            (pl.col("sl_s_time") == 0)
                            | (pl.col("pt_s_time") < pl.col("sl_s_time"))  # 🟢 修正
                        )
                    )
                    .then(pl.col("pt_s_time"))
                    .when(pl.col("sl_s_time") > 0)
                    .then(pl.col("sl_s_time"))
                    .otherwise(
                        pl.col("t1_max_short").cast(pl.Int64)
                    ),  # ←変更: Int64にキャスト
                )
                .with_columns(
                    # マイクロ秒の差分から「実経過時間（分）」を算出して Float32 で保持
                    duration_long=(
                        (pl.col("end_l") - pl.col("t0").cast(pl.Int64))
                        / 1_000_000
                        / 60.0  # ←変更: t0をInt64にキャスト
                    ).cast(pl.Float32),
                    duration_short=(
                        (pl.col("end_s") - pl.col("t0").cast(pl.Int64))
                        / 1_000_000
                        / 60.0  # ←変更: t0をInt64にキャスト
                    ).cast(pl.Float32),
                )
                .select(
                    [
                        "t0",
                        "label_long",
                        "label_short",
                        "duration_long",
                        "duration_short",
                    ]
                )
            )

            # 全データ（M1のすべての足）へ Left Join (非トリガー行は自動的にNullになる)
            final_group_df = (
                bets_df_all.rename({"timestamp": "t0"})
                .join(calculated_df, on="t0", how="left")
                .rename({"t0": "timestamp"})
            )

            # atr_valueはシミュレーターで必須になるためドロップせずに保持する
            # 【重要】シミュレーター用の close と atr_value は残しつつ、不要なゴミを捨てる

            # 存在するカラムのみを削除対象にする（安全なdrop）
            drop_candidates = ["open", "high", "low", atr_col_name]
            actual_drops = [c for c in drop_candidates if c in final_group_df.columns]

            labeled_chunks.append(final_group_df.drop(actual_drops))
        if not labeled_chunks:
            return None
        return pl.concat(labeled_chunks).sort("timestamp")

    # =========================================================================
    # 集計・レポート機能 (双方向対応)
    # =========================================================================

    def _update_label_counts_dual(self, df: pl.DataFrame):
        # ロングの集計
        long_counts = df.group_by("label_long").len()
        for row in long_counts.iter_rows(named=True):
            if row["label_long"] in self.label_counts_long:
                self.label_counts_long[row["label_long"]] += row["len"]

        # ショートの集計
        short_counts = df.group_by("label_short").len()
        for row in short_counts.iter_rows(named=True):
            if row["label_short"] in self.label_counts_short:
                self.label_counts_short[row["label_short"]] += row["len"]

    def _collect_report_data_dual(self, df: pl.DataFrame, current_date: dt.date):
        try:
            required_cols = [
                "timestamp",
                "timeframe",
                "is_trigger",
                "label_long",
                "label_short",
                "duration_long",
                "duration_short",
            ]
            if not all(col in df.columns for col in required_cols):
                return

            # --- [修正] OOM回避: トリガーが発火した行(is_trigger==1)だけをレポート用に抽出 ---
            report_df = (
                df.filter(pl.col("is_trigger") == 1)
                .with_columns(pl.lit(current_date).alias("date"))
                .select(
                    [
                        "timeframe",
                        "label_long",
                        "label_short",
                        "duration_long",
                        "duration_short",
                        "date",
                    ]
                )
            )
            self.report_data.extend(report_df.to_dicts())
        except Exception as e:
            logging.warning(f"Error collecting report data for {current_date}: {e}")

    def _log_final_summary(self):
        total_samples = sum(self.label_counts_long.values())
        if total_samples == 0:
            logging.warning("No samples were processed.")
            return

        long_win = self.label_counts_long.get(1, 0)
        short_win = self.label_counts_short.get(1, 0)

        long_loss = total_samples - long_win
        short_loss = total_samples - short_win

        scale_pos_weight_long = long_loss / long_win if long_win > 0 else 1.0
        scale_pos_weight_short = short_loss / short_win if short_win > 0 else 1.0

        summary = (
            "\n" + "=" * 60 + "\n"
            f"### V5 Dual-Directional Labeling COMPLETED ###\n"
            f"Output Dir: {self.config.output_dir}\n"
            f"  - Total Labeled Triggers (is_trigger=1): {total_samples}\n"
            f"  - Long Win (1): {long_win} / Loss (0): {long_loss}  => `scale_pos_weight_long`: {scale_pos_weight_long:.4f}\n"
            f"  - Short Win (1): {short_win} / Loss (0): {short_loss} => `scale_pos_weight_short`: {scale_pos_weight_short:.4f}\n"
            + "="
            * 60
        )
        logging.info(summary)

    def _generate_report(self):
        logging.info("Generating detailed execution report...")
        cfg = self.config
        report_path = cfg.output_dir / "execution_report_v5_dual.md"

        if not self.report_data:
            report_path.write_text("# Execution Report\n\nNo data generated.")
            return

        try:
            df = pl.from_dicts(self.report_data)
            total = len(df)
            l_win = df.filter(pl.col("label_long") == 1).height
            s_win = df.filter(pl.col("label_short") == 1).height

            # --- Duration統計: 勝ち/負け別に分解 ---
            df_l_win  = df.filter(pl.col("label_long") == 1)["duration_long"]
            df_l_loss = df.filter(pl.col("label_long") == 0)["duration_long"]
            df_s_win  = df.filter(pl.col("label_short") == 1)["duration_short"]
            df_s_loss = df.filter(pl.col("label_short") == 0)["duration_short"]

            avg_dur_l_win  = df_l_win.mean()  or 0.0
            med_dur_l_win  = df_l_win.median() or 0.0
            avg_dur_l_loss = df_l_loss.mean()  or 0.0
            med_dur_l_loss = df_l_loss.median() or 0.0
            avg_dur_s_win  = df_s_win.mean()  or 0.0
            med_dur_s_win  = df_s_win.median() or 0.0
            avg_dur_s_loss = df_s_loss.mean()  or 0.0
            med_dur_s_loss = df_s_loss.median() or 0.0

            daily_activity = (
                df.group_by("date").len().sort("len", descending=True).limit(10)
            )
            daily_table = "| Date | Valid Setup Samples |\n|:---|---:|\n"
            for row in daily_activity.to_dicts():
                daily_table += f"| `{row['date']}` | `{row['len']:,}` |\n"

            report_content = f"""
# Proxy Labeling Engine - Execution Report (V5 Dual-Directional) ⚔️

### 1. Execution Summary
| Item | Value |
|:---|:---|
| **Filter Applied** | `{cfg.get_filter_description()}` |
| **Target Timeframes** | `{TARGET_TIMEFRAMES}` |
| **ATR Filter** | `atr_ratio >= {ATR_RATIO_THRESHOLD}` (ATR Period: {ATR_PERIOD}, Baseline: {ATR_BASELINE_DAYS} day) |
| **Long Rule** | `PT: {RULE_LONG["pt_mult"]}, SL: {RULE_LONG["sl_mult"]}, TD: {RULE_LONG["td"]}` |
| **Short Rule** | `PT: {RULE_SHORT["pt_mult"]}, SL: {RULE_SHORT["sl_mult"]}, TD: {RULE_SHORT["td"]}` |

### 2. Overall Performance
| Metric | Count | Win Rate |
|:---|---:|---:|
| **Total Setups (ATR Ratio >= threshold)** | `{total:,}` | - |
| **Long Profit-Take** | `{l_win:,}` | `{l_win / total:.2%}` |
| **Short Profit-Take** | `{s_win:,}` | `{s_win / total:.2%}` |

### 3. Event Duration Breakdown (Win vs Loss)
| Direction | Outcome | Avg Duration | Median Duration |
|:---|:---|---:|---:|
| **Long** | Win (PT hit) | `{avg_dur_l_win:.1f} min` | `{med_dur_l_win:.1f} min` |
| **Long** | Loss (SL/TD) | `{avg_dur_l_loss:.1f} min` | `{med_dur_l_loss:.1f} min` |
| **Short** | Win (PT hit) | `{avg_dur_s_win:.1f} min` | `{med_dur_s_win:.1f} min` |
| **Short** | Loss (SL/TD) | `{avg_dur_s_loss:.1f} min` | `{med_dur_s_loss:.1f} min` |

### 4. Top 10 Busiest Days (Setups)
{daily_table.strip()}
"""
            report_path.write_text(report_content.strip())
            logging.info(f"Report saved to: {report_path}")
        except Exception as e:
            logging.error(f"Failed to generate report: {e}", exc_info=True)


# =========================================================================
# CLI エントリーポイント
# =========================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="[Phase 3] Create Final V5 Dual-Directional Labels."
    )
    parser.add_argument(
        "--filter-mode", type=str, default=None, choices=["year", "month", "all"]
    )
    parser.add_argument("--year", type=int, default=None)
    parser.add_argument("--year-month", type=str, default=None)
    parser.add_argument("--no-resume", action="store_true")

    args = parser.parse_args()

    filter_year_arg = args.year
    filter_month_arg = None
    filter_mode_arg = args.filter_mode

    # --year-month が指定された場合、自動的に month モードとして処理
    if args.year_month:
        match = re.match(r"(\d{4})/(\d{1,2})$", args.year_month)
        if match:
            filter_year_arg, filter_month_arg = map(int, match.groups())
            filter_mode_arg = "month"

    # --year のみ指定された場合、自動的に year モードとして処理
    elif args.year:
        if not filter_mode_arg:
            filter_mode_arg = "year"

    # どちらも指定がない、かつ明示的な指定もなければ all にフォールバック
    if not filter_mode_arg:
        filter_mode_arg = "all"

    config = ProxyLabelConfig(
        filter_mode=filter_mode_arg,
        filter_year=filter_year_arg,
        filter_month=filter_month_arg,
        resume=not args.no_resume,
    )
    try:
        temp_engine_for_validation = ProxyLabelingEngine(config)
    except ValueError as e:
        print(f"Configuration Error: {e}")
        sys.exit(1)

    print("\nStarting V5 Dual-Directional Labeling Engine...")
    engine = ProxyLabelingEngine(config)
    engine.run()
