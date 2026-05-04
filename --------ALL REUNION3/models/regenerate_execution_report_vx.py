# regenerate_execution_report_vx.py
# =====================================================================
# 既存のS6_LABELED_DATASETから正しいexecution_report_vx_dual.mdを再生成する
#
# 用途：
#   ラベリング実行後にレポート内容が不正確だった場合に
#   データを再実行せずにレポートのみ正しい値で再生成する
#
# 使い方：
#   python regenerate_execution_report_vx.py
# =====================================================================

import sys
import logging
from pathlib import Path
import polars as pl
import datetime as dt

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from blueprint import S6_LABELED_DATASET

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("RegenReport")

# =====================================================================
# 新パラメータ定義（ラベリング時と同じ設定）
# =====================================================================
TARGET_TIMEFRAMES      = ["M1"]
ATR_PERIOD             = 13
ATR_BASELINE_DAYS      = 1
SPREAD                 = 0.50

RULE_LONG = {
    "pt_mult": 10.0,
    "sl_mult": 10.0,
    "td": "120m",
    "atr_ratio_threshold": 3.0,
}
RULE_SHORT = {
    "pt_mult": 10.0,
    "sl_mult": 5.0,
    "td": "120m",
    "atr_ratio_threshold": 1.5,
}

long_atr_th  = RULE_LONG["atr_ratio_threshold"]
short_atr_th = RULE_SHORT["atr_ratio_threshold"]


def main():
    output_dir = S6_LABELED_DATASET
    report_path = output_dir / "execution_report_vx_dual.md"

    logger.info(f"S6_LABELED_DATASETを読み込んでいます: {output_dir}")

    # is_trigger=1のデータのみ読み込む
    required_cols = [
        "timestamp", "timeframe", "atr_ratio",
        "label_long", "label_short",
        "duration_long", "duration_short",
    ]

    lf = (
        pl.scan_parquet(str(output_dir / "**/*.parquet"))
        .filter(pl.col("is_trigger") == 1)
        .select([c for c in required_cols
                 if c in pl.scan_parquet(
                     str(output_dir / "**/*.parquet")
                 ).collect_schema().names()])
    )

    logger.info("データをcollect中...")
    df = lf.collect()
    logger.info(f"is_trigger=1 件数: {len(df)}")

    # =====================================================================
    # 統計計算
    # =====================================================================

    # Short側（ATR>=1.5）: 全is_trigger件数
    total_short = len(df.filter(pl.col("atr_ratio") >= short_atr_th))

    # Long側（ATR>=3.0）: Long eligible件数
    df_long_eligible = df.filter(pl.col("atr_ratio") >= long_atr_th)
    total_long = len(df_long_eligible)

    l_win = int(df.filter(pl.col("label_long") == 1).height)
    s_win = int(df.filter(
        (pl.col("label_short") == 1) & (pl.col("atr_ratio") >= short_atr_th)
    ).height)

    l_wr = l_win / total_long  if total_long > 0 else 0.0
    s_wr = s_win / total_short if total_short > 0 else 0.0

    # Duration統計
    df_l_win  = df.filter(pl.col("label_long")  == 1)["duration_long"]
    df_l_loss = df.filter(pl.col("label_long")  == 0)["duration_long"]
    df_s_win  = df.filter(pl.col("label_short") == 1)["duration_short"]
    df_s_loss = df.filter(pl.col("label_short") == 0)["duration_short"]

    avg_dur_l_win  = df_l_win.mean()   or 0.0
    med_dur_l_win  = df_l_win.median() or 0.0
    avg_dur_l_loss = df_l_loss.mean()  or 0.0
    med_dur_l_loss = df_l_loss.median() or 0.0
    avg_dur_s_win  = df_s_win.mean()   or 0.0
    med_dur_s_win  = df_s_win.median() or 0.0
    avg_dur_s_loss = df_s_loss.mean()  or 0.0
    med_dur_s_loss = df_s_loss.median() or 0.0

    # Top10 Busiest Days
    df_date = df.with_columns(
        pl.col("timestamp").dt.date().alias("date")
    )
    daily_activity = (
        df_date.group_by("date").len()
        .sort("len", descending=True)
        .limit(10)
    )
    daily_table = "| Date | Valid Setup Samples |\n|:---|---:|\n"
    for row in daily_activity.to_dicts():
        daily_table += f"| `{row['date']}` | `{row['len']:,}` |\n"

    # scale_pos_weight
    l_loss = total_long - l_win
    s_loss = total_short - s_win
    spw_long  = l_loss / l_win  if l_win  > 0 else 0.0
    spw_short = s_loss / s_win  if s_win  > 0 else 0.0

    logger.info(f"Long:  {l_win:,}件 / {total_long:,}件 = {l_wr:.2%}")
    logger.info(f"Short: {s_win:,}件 / {total_short:,}件 = {s_wr:.2%}")

    # =====================================================================
    # レポート生成
    # =====================================================================
    report_content = f"""# Proxy Labeling Engine - Execution Report (VX Dual-Directional) ⚔️

*※ このレポートはS6_LABELED_DATASETから逆算して再生成されました（{dt.datetime.now().strftime('%Y-%m-%d %H:%M')}）*

### 1. Execution Summary
| Item | Value |
|:---|:---|
| **Filter Applied** | `All Time` |
| **Target Timeframes** | `{TARGET_TIMEFRAMES}` |
| **Long ATR Filter** | `atr_ratio >= {long_atr_th}` (ATR Period: {ATR_PERIOD}, Baseline: {ATR_BASELINE_DAYS} day) |
| **Short ATR Filter** | `atr_ratio >= {short_atr_th}` (ATR Period: {ATR_PERIOD}, Baseline: {ATR_BASELINE_DAYS} day) |
| **Long Rule** | `PT: {RULE_LONG['pt_mult']}, SL: {RULE_LONG['sl_mult']}, TD: {RULE_LONG['td']}` |
| **Short Rule** | `PT: {RULE_SHORT['pt_mult']}, SL: {RULE_SHORT['sl_mult']}, TD: {RULE_SHORT['td']}` |
| **Label Method** | `net_pnl > 0` (PT・TO問わず純損益がプラスなら勝ち) |
| **Spread** | `{SPREAD}` |

### 2. Overall Performance
| Metric | Count | Win Rate |
|:---|---:|---:|
| **Long Eligible Setups (ATR >= {long_atr_th})** | `{total_long:,}` | - |
| **Short Eligible Setups (ATR >= {short_atr_th})** | `{total_short:,}` | - |
| **Long Profit-Take (net_pnl > 0)** | `{l_win:,}` | `{l_wr:.2%}` |
| **Short Profit-Take (net_pnl > 0)** | `{s_win:,}` | `{s_wr:.2%}` |

### 3. LightGBM scale_pos_weight
| Direction | Win | Loss | scale_pos_weight |
|:---|---:|---:|---:|
| **Long** | `{l_win:,}` | `{l_loss:,}` | `{spw_long:.4f}` |
| **Short** | `{s_win:,}` | `{s_loss:,}` | `{spw_short:.4f}` |

### 4. Event Duration Breakdown (Win vs Loss)
| Direction | Outcome | Avg Duration | Median Duration |
|:---|:---|---:|---:|
| **Long** | Win (net_pnl > 0) | `{avg_dur_l_win:.1f} min` | `{med_dur_l_win:.1f} min` |
| **Long** | Loss (net_pnl <= 0) | `{avg_dur_l_loss:.1f} min` | `{med_dur_l_loss:.1f} min` |
| **Short** | Win (net_pnl > 0) | `{avg_dur_s_win:.1f} min` | `{med_dur_s_win:.1f} min` |
| **Short** | Loss (net_pnl <= 0) | `{avg_dur_s_loss:.1f} min` | `{med_dur_s_loss:.1f} min` |

### 5. Top 10 Busiest Days (Setups)
{daily_table.strip()}
"""

    report_path.write_text(report_content.strip())
    logger.info(f"レポート保存: {report_path}")
    print(f"\n✅ 完了: {report_path}")


if __name__ == "__main__":
    main()
