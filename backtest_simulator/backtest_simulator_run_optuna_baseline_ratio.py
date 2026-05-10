# /workspace/backtest_simulator/backtest_simulator_run_optuna_baseline_ratio.py
#
# 【目的】 baseline_ratio（相対ボラ比率）の最適閾値をOptunaで探索する
#
# 【baseline_ratio の定義】
#   baseline_ratio = mean(ATR, 1日=480本) / mean(ATR, N日=N×480本)
#                  = 「昨日のボラ」vs「過去N日平均のボラ」の相対比率
#   → XAU $1800でも$4600でもスケール消去 → 普遍的フィルター
#
# 【絶対床との違い】
#   min_baseline_atr (絶対): XAU価格水準が上がると陳腐化する可能性あり
#   min_baseline_ratio(相対): 価格水準依存なし → 将来も有効
#
# 【探索設計】
#   探索①: min_baseline_ratio [0.0, 0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
#   探索②: min_atr_threshold  [0.80, 0.90]
#   固定:  min_baseline_atr=0.0（相対フィルター単体の効果を純粋に測定）
#          baseline_ratio_lookback_days=7（1週間比較）
#          m2_th=0.70, m2_delta=0.30, risk=2%
#   総Trial: 9 × 2 = 18

import sys
import pickle
from pathlib import Path
import logging
import optuna
import pandas as pd

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from blueprint import (  # noqa: E402
    S7_BACKTEST_OPTUNA_RESULTS,
    S7_BACKTEST_CACHE,
    S7_M2_OOF_PREDICTIONS_LONG,
    S7_M2_OOF_PREDICTIONS_SHORT,
)
from backtest_simulator_cimera import BacktestConfig, BacktestSimulator  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [OPTUNA_RATIO] %(levelname)s - %(message)s",
)

OPTUNA_TEST_LIMIT  = 0
MAX_ALLOWED_DRAWDOWN = 0.0
MIN_TRADES_THRESHOLD = 100

global_preloaded_data = None


def objective(trial: optuna.Trial):
    # =========================================================
    # 1. 探索パラメータ
    # =========================================================
    # [探索①] baseline_ratio: 昨日ボラ / 過去7日ボラ の下限
    min_baseline_ratio = trial.suggest_categorical(
        "min_baseline_ratio",
        [0.0, 0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90],
    )

    # [探索②] ATR Ratio閾値（現行0.8 vs 0.9）
    min_atr_threshold = trial.suggest_categorical(
        "min_atr_threshold",
        [0.80, 0.90],
    )

    # =========================================================
    # 2. 固定パラメータ
    # =========================================================
    m2_th          = 0.70
    m2_delta       = 0.30
    fixed_risk_pct = 0.02

    # --- 絶対床は無効（相対フィルター単体の効果を純粋に測定）---
    min_baseline_atr = 0.0
    # min_baseline_atr = 0.90  # 絶対床との組み合わせを試す場合はここを変更

    # --- lookback日数 ---
    lookback_days = 7
    # lookback_days = 14  # 14日比較を試す場合はここを変更

    # --- 旧探索パラメータ（参考記録として保持）---
    # m2_th = trial.suggest_categorical("m2_th", [0.30, 0.50, 0.70, 0.80, 0.90, 0.95])
    # m2_delta = trial.suggest_categorical("m2_delta", [0.00, 0.30, 0.50])
    # fixed_risk_pct = trial.suggest_categorical("fixed_risk_pct", [0.02, 0.05, 0.10])

    # =========================================================
    # 3. BacktestConfig
    # =========================================================
    config = BacktestConfig(
        use_fixed_risk=True,
        fixed_risk_percent=fixed_risk_pct,
        m2_proba_threshold=m2_th,
        m2_delta_threshold=m2_delta,
        min_atr_threshold=min_atr_threshold,
        min_baseline_atr=min_baseline_atr,
        min_baseline_ratio=min_baseline_ratio,          # ★ 相対フィルター
        baseline_ratio_lookback_days=lookback_days,     # ★ 7日比較
        max_positions=100,
        max_consecutive_sl=2,
        cooldown_minutes_after_sl=30,
        test_limit_partitions=OPTUNA_TEST_LIMIT,
        oof_mode=True,
        sl_multiplier_long=5.0,
        pt_multiplier_long=1.0,
        sl_multiplier_short=5.0,
        pt_multiplier_short=1.0,
        td_minutes_long=30.0,
        td_minutes_short=30.0,
        margin_call_percent=0.0,
        stop_out_percent=0.0,
    )

    logging.info(f"--- Trial {trial.number:03d} ---")
    logging.info(
        f"baseline_ratio>={min_baseline_ratio}(lookback={lookback_days}d) "
        f"atr_ratio>={min_atr_threshold} | "
        f"M2={m2_th} Delta={m2_delta} Risk={fixed_risk_pct*100:.0f}%"
    )

    logger = logging.getLogger()
    orig = logger.level
    logger.setLevel(logging.WARNING)
    try:
        sim = BacktestSimulator(config)
        report_data = sim.run(preloaded_data=global_preloaded_data)
    except Exception as e:
        logger.setLevel(orig)
        logging.error(f"Trial {trial.number:03d} failed: {e}")
        raise optuna.exceptions.TrialPruned()
    logger.setLevel(orig)

    # =========================================================
    # 4. 評価指標
    # =========================================================
    max_dd        = report_data.get("max_drawdown_pct", 0.0) or 0.0
    sharpe        = report_data.get("sharpe_ratio_annual", 0.0) or 0.0
    profit_factor = report_data.get("profit_factor", 0.0) or 0.0
    trades        = report_data.get("total_trades", 0) or 0
    win_rate      = report_data.get("win_rate_pct", 0.0) or 0.0
    min_margin    = report_data.get("min_margin_level_pct", 9999.0)
    stop_outs     = report_data.get("stop_out_count", 0) or 0
    net_profit    = report_data.get("final_capital", 0.0) - report_data.get("initial_capital", 0.0)

    trial.set_user_attr("Total_Net_Profit",   net_profit)
    trial.set_user_attr("Max_DD_Pct",         max_dd)
    trial.set_user_attr("Sharpe_Ratio",       sharpe)
    trial.set_user_attr("Profit_Factor",      profit_factor)
    trial.set_user_attr("Win_Rate_Pct",       win_rate)
    trial.set_user_attr("Total_Trades",       trades)
    trial.set_user_attr("Min_Margin_Pct",     float(min_margin) if min_margin != 9999.0 else None)

    if trades < MIN_TRADES_THRESHOLD:
        raise optuna.exceptions.TrialPruned()
    if stop_outs > 0:
        return -9999.0
    if max_dd > MAX_ALLOWED_DRAWDOWN:
        return -9999.0
    if min_margin < 150.0:
        sharpe -= (150.0 - min_margin) * 10

    score = sharpe + (profit_factor * 0.1)
    logging.info(
        f"Trial {trial.number:03d} Done: Score={score:.4f} "
        f"PF={profit_factor:.2f} Sharpe={sharpe:.2f} Trades={trades} DD={max_dd:.1f}%"
    )
    return float(score)


def export_rankings(study: optuna.Study):
    rows = []
    for t in study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE: continue
        if t.value is None or t.value == -9999.0: continue
        rows.append({
            "Trial_ID":           t.number,
            "Score":              t.value,
            "min_baseline_ratio": t.params.get("min_baseline_ratio"),
            "min_atr_threshold":  t.params.get("min_atr_threshold"),
            "lookback_days_fixed": 7,
            "m2_th_fixed":        0.70,
            "m2_delta_fixed":     0.30,
            "risk_pct_fixed":     0.02,
            "Sharpe_Ratio":       t.user_attrs.get("Sharpe_Ratio"),
            "Profit_Factor":      t.user_attrs.get("Profit_Factor"),
            "Win_Rate_Pct":       t.user_attrs.get("Win_Rate_Pct"),
            "Max_DD_Pct":         t.user_attrs.get("Max_DD_Pct"),
            "Total_Net_Profit":   t.user_attrs.get("Total_Net_Profit"),
            "Total_Trades":       t.user_attrs.get("Total_Trades"),
        })
    if not rows:
        logging.warning("No complete trials.")
        return

    df = pd.DataFrame(rows).sort_values("Profit_Factor", ascending=False)
    out = S7_BACKTEST_OPTUNA_RESULTS / "ranking_baseline_ratio_optuna.csv"
    df.to_csv(out, index=False)

    print(f"\n{'='*85}")
    print("  🏆 Ranking — baseline_ratio (relative) × ATR_Ratio Optuna")
    print(f"{'='*85}")
    print(f"{'#':<4} {'Trial':>6} {'ratio_thr':>10} {'atr_thr':>8} "
          f"{'PF':>7} {'Sharpe':>7} {'Win%':>7} {'Trades':>7} {'DD%':>7}")
    print("-"*85)
    for rank, (_, r) in enumerate(df.iterrows(), 1):
        print(f"{rank:<4} {int(r['Trial_ID']):>6} "
              f"{r['min_baseline_ratio']:>10} {r['min_atr_threshold']:>8} "
              f"{r['Profit_Factor']:>7.2f} {r['Sharpe_Ratio']:>7.2f} "
              f"{r['Win_Rate_Pct']:>7.2f} {int(r['Total_Trades']):>7} "
              f"{r['Max_DD_Pct']:>7.2f}")


def save_csv_callback(study, trial):
    export_rankings(study)


if __name__ == "__main__":
    S7_BACKTEST_OPTUNA_RESULTS.mkdir(parents=True, exist_ok=True)

    def load_or_generate_cache():
        if S7_BACKTEST_CACHE.exists():
            cache_mtime = S7_BACKTEST_CACHE.stat().st_mtime
            oof_mtime = max(
                S7_M2_OOF_PREDICTIONS_LONG.stat().st_mtime,
                S7_M2_OOF_PREDICTIONS_SHORT.stat().st_mtime,
            )
            stale = cache_mtime < oof_mtime
            print(f"\nキャッシュ: {S7_BACKTEST_CACHE}")
            if stale:
                print("  ⚠️  キャッシュがOOFより古い可能性があります。")
            print("  [y] 使用  [r] 再生成")
            ans = input("選択 [y/r]: ").strip().lower()
            if ans != "r":
                with open(S7_BACKTEST_CACHE, "rb") as f:
                    data = pickle.load(f)
                # [baseline_ratio列の存在チェック]
                # 旧キャッシュには baseline_ratio 列がないため、
                # そのまま使うとフィルターが無音でスキップされ全Trial同一結果になる。
                # 先頭チャンクで列の有無を確認し、なければ強制再生成する。
                sample_chunk = next(iter(data[0].values()))
                if "baseline_ratio" not in sample_chunk.columns:
                    print()
                    print("  ❌ このキャッシュには baseline_ratio 列がありません。")
                    print("     旧バージョンのキャッシュです。再生成が必要です。")
                    print("     削除して再生成します...")
                    S7_BACKTEST_CACHE.unlink()
                    # 再生成フローへ fall-through
                else:
                    logging.info("キャッシュ読み込み完了 (baseline_ratio列確認済み)。")
                    return data
            S7_BACKTEST_CACHE.unlink()

        logging.info("データ生成中...")
        dummy = BacktestSimulator(BacktestConfig(
            test_limit_partitions=OPTUNA_TEST_LIMIT,
            baseline_ratio_lookback_days=7,  # キャッシュ生成時も7日で
        ))
        data = dummy.preload_data()
        with open(S7_BACKTEST_CACHE, "wb") as f:
            pickle.dump(data, f)
        logging.info(f"キャッシュ保存: {S7_BACKTEST_CACHE}")
        return data

    global_preloaded_data = load_or_generate_cache()
    logging.info("Preload complete. Starting Optuna...")

    db_path = str(S7_BACKTEST_OPTUNA_RESULTS / "optuna_baseline_ratio.db")
    search_space = {
        "min_baseline_ratio": [0.0, 0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90],
        "min_atr_threshold":  [0.80, 0.90],
    }
    study_name = "baseline_ratio_search_v1"

    study = optuna.create_study(
        study_name=study_name,
        storage=f"sqlite:///{db_path}",
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.GridSampler(search_space),
    )

    n_trials = 18  # 9 × 2
    logging.info(f"Study: {study_name} | {n_trials} trials (9×2 grid)")
    logging.info("baseline_ratio = mean(ATR,1d) / mean(ATR,7d) | fixed: m2=0.70 delta=0.30 risk=2%")
    study.optimize(objective, n_trials=n_trials, callbacks=[save_csv_callback])

    print("\n\n" + "="*85)
    print("  📊 FINAL — baseline_ratio Optimization")
    print("="*85)
    export_rankings(study)

    best = study.best_trial
    print(f"\n  Best Trial: #{best.number:03d}")
    print(f"  Best Score: {study.best_value:.4f}")
    for k, v in best.params.items():
        print(f"    {k}: {v}")
    print(f"  Best PF:    {best.user_attrs.get('Profit_Factor', 'N/A'):.2f}")
    print(f"  Best Trades:{best.user_attrs.get('Total_Trades', 'N/A')}")
    print("="*85)
