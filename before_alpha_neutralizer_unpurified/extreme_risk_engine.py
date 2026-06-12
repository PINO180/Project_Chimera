# /workspace/execution/extreme_risk_engine.py
"""
極限リスク管理エンジン 5.0 (Project Cimera V5仕様)
Two-Brainアーキテクチャ対応: AI推論を外部化し、純粋な資金管理と動的バリア計算に特化

[V5リファクタリング内容]
- 廃止: ケリー基準、モデルの内部ロード・直列推論、時間/曜日フィルター
- 新造: 資金比例固定ロット (Auto Lot) と Exness動的レバレッジ制限
- 新造: ATRベースの純粋なSL/TP計算

[修正履歴]
- [FIX-2] MarketInfo TypedDict を追加 (main.py の ImportError 解消)
- [FIX-3] _get_exness_leverage のレバレッジ閾値をシミュレーターと統一
- [FIX-5] Decimal 精度を 50 → 28 桁に削減
"""

import sys
import json
import logging
from pathlib import Path
from decimal import Decimal, getcontext
from typing import Dict, Any, Optional

try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict

from datetime import datetime

sys.path.append(str(Path(__file__).resolve().parents[1]))
import blueprint as config

# [FIX-5] Decimal の精度を 28 桁に削減 (過剰な 50 桁から最適化)
getcontext().prec = 28

logger = logging.getLogger(__name__)


# [FIX-2] MarketInfo TypedDict を追加 — main.py の ImportError を解消
class MarketInfo(TypedDict, total=False):
    """リスクエンジンに渡す市場文脈情報"""

    atr_value: float
    current_price: float
    # ▼ 修正: 古い pt_multiplier / sl_multiplier を廃止し Long/Short 分割に変更
    sl_multiplier_long: float
    pt_multiplier_long: float
    sl_multiplier_short: float
    pt_multiplier_short: float
    payoff_ratio: float
    direction: int


class ExtremeRiskEngineV5:
    """
    極限リスク管理エンジン 5.0 (Project Cimera V5仕様)
    外部から受け取った確率(p_long/p_short)と純粋な数式に基づく執行管理
    """

    def __init__(self, config_path: str = str(config.CONFIG_RISK)):
        self.config = self._load_config(config_path)
        logger.info("ExtremeRiskEngineV5 (Two-Brain Mode) を初期化しました。")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        try:
            with open(config_path, "r") as f:
                loaded_config = json.load(f)
                logger.info(
                    f"設定ファイル '{Path(config_path).name}' を読み込みました。"
                )
                return loaded_config
        except Exception as e:
            logger.warning(
                f"設定の読み込みに失敗しました。V5デフォルトを使用します: {e}"
            )
            return {
                "base_capital": 1000.0,
                "lot_per_base": 0.1,
                "max_lot_absolute": 200.0,
                "contract_size": 100.0,
                "min_lot_size": 0.01,
                "base_leverage": 2000.0,
                "spread_pips": 36.0,  # 16.0 -> 36.0 (Trial 0同期)
                "value_per_pip": 1.0,
                "min_atr_threshold": 0.8,  # ATR Ratio閾値 (絶対値 2.0 → Ratio 0.8)
                # ▼ 修正: デフォルト値も Long/Short 分割に変更
                "sl_multiplier_long": 5.0,
                "pt_multiplier_long": 1.0,
                "sl_multiplier_short": 5.0,
                "pt_multiplier_short": 1.0,
                "m2_proba_threshold": 0.50,
                "max_drawdown": 0.50,
                "max_positions": 100,  # 1000 -> 100 (Trial 0同期)
                "prevent_simultaneous_orders": True,
                "max_consecutive_sl": 2,
                "cooldown_minutes_after_sl": 30,  # 10 -> 30 (Trial 0同期)
                "fixed_risk_percent": 0.05,  # 追加 (Trial 0同期)
            }

    def _get_exness_leverage(self, equity: Decimal) -> Decimal:
        """
        Exnessの証拠金レベルに基づく動的レバレッジ制限

        [FIX-3] バックテストシミュレーターの _get_effective_leverage() と完全統一:
          equity < $5,000    → base_leverage そのまま (上限なし扱い)
          equity < $30,000   → 2000倍上限
          equity < $100,000  → 1000倍上限
          equity >= $100,000 → 500倍上限
        """
        base_leverage = Decimal(str(self.config.get("base_leverage", 2000.0)))

        if equity < Decimal("5000"):
            limit = base_leverage
        elif equity < Decimal("30000"):
            limit = Decimal("2000")
        elif equity < Decimal("100000"):
            limit = Decimal("1000")
        else:
            limit = Decimal("500")

        return base_leverage.min(limit)

    def calculate_auto_lot(
        self,
        equity: float,
        current_price: float,
        base_capital: Optional[float] = None,
        lot_per_base: Optional[float] = None,
        effective_leverage_cap: Optional[float] = None,
        atr_value: float = 0.0,
        current_spread_pips: Optional[float] = None,
        sl_multiplier: float = 5.0,  # ▼追加: 呼び出し元から必ずSL倍率を受け取る
    ) -> float:
        """
        資金比例固定ロット (Auto Lot) を計算し、レバレッジ上限で安全にキャップする
        [V5改修] スプレッドとATRを加味した厳密なFixed Riskロジックを統合
        """
        eq_dec = Decimal(str(max(equity, 0.0)))
        price_dec = Decimal(str(current_price))
        b_cap_dec = Decimal(str(base_capital or self.config["base_capital"]))
        l_base_dec = Decimal(str(lot_per_base or self.config["lot_per_base"]))
        contract_dec = Decimal(str(self.config["contract_size"]))
        min_lot_dec = Decimal(str(self.config["min_lot_size"]))
        max_lot_abs = Decimal(str(self.config["max_lot_absolute"]))

        if eq_dec <= 0 or price_dec <= 0 or b_cap_dec <= 0:
            return 0.0

        use_fixed_risk = self.config.get("use_fixed_risk", True)

        if use_fixed_risk and atr_value > 0.0:
            # --- 厳密な固定比率（Fixed Risk）---
            fixed_risk_percent = Decimal(
                str(
                    self.config.get("fixed_risk_percent", 0.05)
                )  # デフォルトを0.05に変更
            )
            # ▼修正: コンフィグを直接見に行くのではなく、引数として受け取った正確な倍率を使用する
            sl_mult = Decimal(str(sl_multiplier))

            # 1. 最大許容損失額 = 現在の口座残高 × fixed_risk_percent
            max_loss_amount = eq_dec * fixed_risk_percent

            # 2. 1ロットあたりの値幅損失額 = ATR × SL_Multiplier × ContractSize
            # [乖離②修正] バックテスト側と計算式を統一:
            #   バックテスト: base_lot = max_loss / (sl_price_distance * contract_size)
            #   旧本番: base_lot = max_loss / (sl_loss + spread_cost)  ← スプレッドを分母に含めていた
            #   新本番: base_lot = max_loss / sl_loss  ← バックテストと一致
            sl_loss_per_lot = Decimal(str(atr_value)) * sl_mult * contract_dec

            # 3. base_lot = 最大許容損失額 / 値幅損失額
            if sl_loss_per_lot > Decimal("0"):
                base_lot = max_loss_amount / sl_loss_per_lot
            else:
                base_lot = Decimal("0")
        else:
            # --- 従来の固定複利（Auto Lot）---
            base_lot = (eq_dec / b_cap_dec) * l_base_dec
        # ▲▲▲ ここまで修正 ▲▲▲

        if effective_leverage_cap is not None:
            leverage_dec = Decimal(str(effective_leverage_cap))
        else:
            leverage_dec = self._get_exness_leverage(eq_dec)

        max_lot_margin = (eq_dec * leverage_dec) / (price_dec * contract_dec)
        final_lot = min(base_lot, max_lot_margin, max_lot_abs)
        final_lot_quantized = float((final_lot // min_lot_dec) * min_lot_dec)

        # ▼▼▼ 新規追加: ニート化防止パッチ (最低ロットの強制保証) ▼▼▼
        final_lot_quantized = max(final_lot_quantized, float(min_lot_dec))

        logger.debug(
            f"Lot計算(FixedRisk={use_fixed_risk}): 基本={base_lot:.2f}, 証拠金上限({leverage_dec}倍)={max_lot_margin:.2f} -> 最終={final_lot_quantized:.2f}"
        )
        return final_lot_quantized

    def calculate_sl_tp(
        self,
        entry_price: float,
        atr: float,
        direction: int,
        sl_multiplier: float,
        tp_multiplier: float,
    ) -> Dict[str, float]:
        """ATR値を絶対の定規とした純粋なSL/TP計算"""
        if direction == 1:
            sl = entry_price - (atr * sl_multiplier)
            tp = entry_price + (atr * tp_multiplier)
        elif direction == -1:
            sl = entry_price + (atr * sl_multiplier)
            tp = entry_price - (atr * tp_multiplier)
        else:
            raise ValueError(
                f"無効な取引方向です: {direction} (1 または -1 を指定してください)"
            )

        return {"stop_loss": round(sl, 3), "take_profit": round(tp, 3)}

    def generate_trade_command(
        self,
        action: str,
        p_long: float,
        p_short: float,
        current_price: float,
        atr: float,
        equity: float,
        sl_multiplier: Optional[float] = None,
        tp_multiplier: Optional[float] = None,
        base_capital: Optional[float] = None,
        lot_per_base: Optional[float] = None,
        current_spread_pips: Optional[float] = None,
        atr_ratio: Optional[float] = None,  # ★追加: realtime_feature_engineから渡されるATR Ratio
    ) -> Dict[str, Any]:
        """外部(司令塔)から受け取った確率とパラメータに基づき、最終的な執行コマンドを生成する"""

        # ▼修正: 未指定時は Long/Short の方向に応じて正しいコンフィグキーから取得
        if sl_multiplier is None:
            sl_mult = self.config.get(
                "sl_multiplier_long" if action == "BUY" else "sl_multiplier_short", 5.0
            )
        else:
            sl_mult = sl_multiplier

        if tp_multiplier is None:
            tp_mult = self.config.get(
                "pt_multiplier_long" if action == "BUY" else "pt_multiplier_short", 1.0
            )
        else:
            tp_mult = tp_multiplier

        if action == "HOLD":
            return self._generate_hold_command("司令塔からのHOLD指示", p_long, p_short)

        # 追加: ATR Ratioフィルター (ボラティリティ枯渇時の発注スキップ)
        # atr_ratioはrealtime_feature_engineのmarket_infoから渡されたものを使用
        min_atr = self.config.get("min_atr_threshold", 0.8)  # ATR Ratio閾値
        effective_atr_ratio = atr_ratio if atr_ratio is not None else 1.0
        if effective_atr_ratio < min_atr:
            return self._generate_hold_command(
                f"ATR Ratio({effective_atr_ratio:.3f})が最小閾値({min_atr})未満のためスキップ",
                p_long,
                p_short,
            )

        spread_val = (
            current_spread_pips
            if current_spread_pips is not None
            else self.config.get("spread_pips", 36.0)  # デフォルト16.0 -> 36.0
        )
        max_spread = self.config.get(
            "max_allowed_spread", 50.0
        )  # デフォルト30.0 -> 50.0
        if spread_val > max_spread:
            return self._generate_hold_command(
                f"スプレッド({spread_val:.1f}pips)が許容上限({max_spread:.1f}pips)を超過したため発注をブロック",
                p_long,
                p_short,
            )

        direction = 1 if action == "BUY" else -1

        lots = self.calculate_auto_lot(
            equity=equity,
            current_price=current_price,
            base_capital=base_capital,
            lot_per_base=lot_per_base,
            atr_value=atr,
            current_spread_pips=spread_val,
            sl_multiplier=sl_mult,  # ▼修正: 確定した SL 倍率をロット計算関数にパスする！
        )

        sl_tp = self.calculate_sl_tp(
            entry_price=current_price,
            atr=atr,
            direction=direction,
            sl_multiplier=sl_mult,  # 修正反映
            tp_multiplier=tp_mult,  # 修正反映
        )

        # [SL/TP-FIX] MQL5側でOrderSend直前のAsk/Bid基準にSL/TPを再計算するため、
        # ドル幅(sl_width/tp_width)・ATR・乗数をコマンドに追加する。
        # Python送信時とMT5約定時の価格差(スリッページ)があっても、
        # ポジションから見たSL/TP幅は常に意図通り(atr*sl_mult / atr*tp_mult)となる。
        # MQL5側ではキー不在時に0.0が返るため、フォールバックでstop_loss/take_profitの絶対価格を使用する。
        sl_width = float(atr) * float(sl_mult)
        tp_width = float(atr) * float(tp_mult)

        trade_command = {
            "action": action,
            "lots": lots,
            "entry_price": current_price,
            "stop_loss": sl_tp["stop_loss"],
            "take_profit": sl_tp["take_profit"],
            # [SL/TP-FIX] MQL5側でAsk/Bid基準のSL/TP再計算に使用
            "sl_width": sl_width,
            "tp_width": tp_width,
            "atr": float(atr),
            "sl_multiplier": float(sl_mult),
            "tp_multiplier": float(tp_mult),
            "p_long": p_long,
            "p_short": p_short,
            "reason": f"TwoBrain Signal: {action} (p_long={p_long:.2f}, p_short={p_short:.2f})",
            "timestamp": datetime.now().isoformat(),
        }

        logger.info(
            f"✓ コマンド生成: {action} {lots}Lot @ {current_price:.3f} (SL: {sl_tp['stop_loss']:.3f}, TP: {sl_tp['take_profit']:.3f})"
        )
        return trade_command

    def _generate_hold_command(
        self, reason: str, p_long: float, p_short: float
    ) -> Dict[str, Any]:
        return {
            "action": "HOLD",
            "lots": 0.0,
            "entry_price": 0.0,
            "stop_loss": 0.0,
            "take_profit": 0.0,
            "p_long": p_long,
            "p_short": p_short,
            "reason": reason,
            "timestamp": datetime.now().isoformat(),
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    engine = ExtremeRiskEngineV5()
    print(
        engine.generate_trade_command(
            action="BUY",
            p_long=0.85,
            p_short=0.15,
            current_price=2350.50,
            atr=5.2,
            equity=50000.0,
            sl_multiplier=2.0,
            tp_multiplier=3.0,
        )
    )
