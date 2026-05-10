# /workspace/backtest_simulator/backtest_simulator_run_optuna_baseline.py
#
# 【目的】 M2確信度閾値 / Delta の2パラメータをOptunaで探索する
# 【探索】
#   m2_th    : [0.70, 0.80]          2値
#   m2_delta : [0.30, 0.50, 0.80]    3値
#   総Trial  : 2 × 3 = 6
# 【固定】
#   min_baseline_atr  = 0.0   (フィルターなし)
#   min_atr_threshold = 0.90  (分析結果より)
#   fixed_risk_pct    = 0.02

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
from backtest_simulator_cimera import (  # noqa: E402
    BacktestConfig,
    BacktestSimulator,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [OPTUNA_BASELINE] %(levelname)s - %(message)s",
)

# ==========================================
# オプティマイザーの全体設定
# ==========================================
OPTUNA_TEST_LIMIT    = 0
MAX_ALLOWED_DRAWDOWN = 0.0
MIN_TRADES_THRESHOLD = 100
# ==========================================

global_preloaded_data = None


def objective(trial: optuna.Trial):
    # =========================================================
    # 1. 探索パラメータ
    # =========================================================

    # [探索①] M2確信度閾値
    m2_th = trial.suggest_categorical(
        "m2_th",
        [0.70, 0.80],
    )

    # [探索②] Delta閾値
    m2_delta = trial.suggest_categorical(
        "m2_delta",
        [0.30, 0.50, 0.80],
    )

    # =========================================================
    # 2. 固定パラメータ
    # =========================================================

    # --- 固定: baseline絶対床 ---
    min_baseline_atr = 0.0
    # min_baseline_atr = trial.suggest_categorical(
    #     "min_baseline_atr",
    #     [0.0, 0.70, 0.75, 0.80, 0.82, 0.85,
    #      0.90, 1.00, 1.10],
    # )

    # --- 固定: ATR Ratio閾値 ---
    min_atr_threshold = 0.90
    # min_atr_threshold = trial.suggest_categorical(
    #     "min_atr_threshold",
    #     [0.80,
    #      0.90],
    # )

    # --- 固定: リスク比率 ---
    fixed_risk_pct = 0.02
    # fixed_risk_pct = trial.suggest_categorical(
    #     "fixed_risk_pct",
    #     [0.02, 0.05, 0.10],
    # )

    # --- 相対系フィルターは無効 ---
    min_baseline_ratio = 0.0
    min_sar_threshold  = 0.0

    # =========================================================
    # 3. BacktestConfigの構築
    # =========================================================
    config = BacktestConfig(
        use_fixed_risk=True,
        fixed_risk_percent=fixed_risk_pct,
        m2_proba_threshold=m2_th,
        m2_delta_threshold=m2_delta,
        min_atr_threshold=min_atr_threshold,
        min_baseline_atr=min_baseline_atr,
        min_baseline_ratio=min_baseline_ratio,
        min_sar_threshold=min_sar_threshold,
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

    logging.info(f"--- Starting Trial {trial.number:03d} ---")
    logging.info(
        f"m2_th={m2_th} delta={m2_delta} "
        f"| ratio={min_atr_threshold} baseline={min_baseline_atr} risk={fixed_risk_pct*100:.0f}%"
    )

    logger = logging.getLogger()
    original_level = logger.level
    logger.setLevel(logging.WARNING)
    try:
        simulator = BacktestSimulator(config)
        report_data = simulator.run(preloaded_data=global_preloaded_data)
    except Exception as e:
        logger.setLevel(original_level)
        logging.error(f"Trial {trial.number:03d} failed: {e}")
        raise optuna.exceptions.TrialPruned()
    logger.setLevel(original_level)

    # =========================================================
    # 4. 出力フォルダ（中身は最良Trial確認後に手動取得）
    # =========================================================
    folder_name = (
        f"Trial_{trial.number:03d}"
        f"_th{m2_th}"
        f"_d{m2_delta}"
    )
    dst_dir = S7_BACKTEST_OPTUNA_RESULTS / folder_name
    dst_dir.mkdir(parents=True, exist_ok=True)

    # ファイルコピーはコメントアウト（最良Trial確認後に手動実施）
    # for fname in [
    #     "final_backtest_report_v5_M2.json",
    #     "final_backtest_report_v5_M2.txt",
    #     "equity_curve_v5_M2.png",
    #     "detailed_trade_log_v5_M2.csv",
    # ]:
    #     src_file = S7_BACKTEST_SIM_RESULTS / fname
    #     if src_file.exists():
    #         shutil.copy2(src_file, dst_dir / fname)

    # =========================================================
    # 5. 評価指標の抽出 & Trial属性セット
    # =========================================================
    max_dd        = report_data.get("max_drawdown_pct", 0.0) or 0.0
    sharpe        = report_data.get("sharpe_ratio_annual", 0.0) or 0.0
    # sortino     = report_data.get("sortino_ratio_annual", 0.0) or 0.0
    profit_factor = report_data.get("profit_factor", 0.0) or 0.0
    trades        = report_data.get("total_trades", 0) or 0
    win_rate      = report_data.get("win_rate_pct", 0.0) or 0.0
    min_margin    = report_data.get("min_margin_level_pct", 9999.0)
    stop_outs     = report_data.get("stop_out_count", 0) or 0
    net_profit    = (
        report_data.get("final_capital", 0.0)
        - report_data.get("initial_capital", 0.0)
    )

    trial.set_user_attr("Total_Net_Profit",  net_profit)
    trial.set_user_attr("Max_DD_Pct",        max_dd)
    trial.set_user_attr("Sharpe_Ratio",      sharpe)
    trial.set_user_attr("Profit_Factor",     profit_factor)
    trial.set_user_attr("Win_Rate_Pct",      win_rate)
    trial.set_user_attr("Total_Trades",      trades)
    trial.set_user_attr(
        "Min_Margin_Pct",
        float(min_margin) if min_margin != 9999.0 else None,
    )

    # =========================================================
    # 6. ペナルティ判定 → スコア算出
    # =========================================================
    if trades < MIN_TRADES_THRESHOLD:
        logging.info(f"Trial {trial.number:03d} Pruned: trades={trades}")
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
        f"(Sharpe={sharpe:.2f} PF={profit_factor:.2f} "
        f"DD={max_dd:.1f}% Trades={trades})"
    )
    return float(score)


# =========================================================
# ランキング出力
# =========================================================
def export_rankings(study: optuna.Study):
    rows = []
    for t in study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue
        if t.value is None or t.value == -9999.0:
            continue
        rows.append({
            "Trial_ID":           t.number,
            "Score":              t.value,
            "m2_th":              t.params.get("m2_th"),
            "m2_delta":           t.params.get("m2_delta"),
            "min_atr_threshold_fixed":  0.90,
            "min_baseline_atr_fixed":   0.0,
            "risk_pct_fixed":           0.02,
            "Sharpe_Ratio":       t.user_attrs.get("Sharpe_Ratio"),
            "Profit_Factor":      t.user_attrs.get("Profit_Factor"),
            "Win_Rate_Pct":       t.user_attrs.get("Win_Rate_Pct"),
            "Max_DD_Pct":         t.user_attrs.get("Max_DD_Pct"),
            "Total_Net_Profit":   t.user_attrs.get("Total_Net_Profit"),
            "Total_Trades":       t.user_attrs.get("Total_Trades"),
            "Min_Margin_Pct":     t.user_attrs.get("Min_Margin_Pct"),
        })

    if not rows:
        logging.warning("No complete trials.")
        return

    df = pd.DataFrame(rows).sort_values("Profit_Factor", ascending=False)
    out = S7_BACKTEST_OPTUNA_RESULTS / "ranking_baseline_atr_optuna_v2.csv"
    df.to_csv(out, index=False)

    print(f"\n{'='*85}")
    print("  🏆 Ranking — M2_th × Delta  (ratio=0.90固定, baseline=0.0固定)")
    print(f"{'='*85}")
    print(
        f"{'#':<4} {'Trial':>6} {'m2_th':>6} {'delta':>6} "
        f"{'PF':>7} {'Sharpe':>7} {'Win%':>7} {'Trades':>7} {'DD%':>7} {'総利益B':>9}"
    )
    print("-"*85)
    for rank, (_, r) in enumerate(df.iterrows(), 1):
        print(
            f"{rank:<4} {int(r['Trial_ID']):>6} "
            f"{r['m2_th']:>6} {r['m2_delta']:>6} "
            f"{r['Profit_Factor']:>7.2f} {r['Sharpe_Ratio']:>7.3f} "
            f"{r['Win_Rate_Pct']:>7.2f} {int(r['Total_Trades']):>7,} "
            f"{r['Max_DD_Pct']:>7.2f} {r['Total_Net_Profit']/1e9:>9.3f}"
        )


def save_csv_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial):
    export_rankings(study)


if __name__ == "__main__":
    S7_BACKTEST_OPTUNA_RESULTS.mkdir(parents=True, exist_ok=True)

    # =========================================================
    # キャッシュ管理
    # =========================================================
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
                print("  ⚠️  OOFより古い可能性があります。")
            print("  [y] 使用  [r] 再生成")
            ans = input("選択 [y/r]: ").strip().lower()
            if ans != "r":
                logging.info("キャッシュ読み込み中...")
                with open(S7_BACKTEST_CACHE, "rb") as f:
                    data = pickle.load(f)
                logging.info("キャッシュ読み込み完了。")
                return data
            S7_BACKTEST_CACHE.unlink()

        logging.info("データ生成中...")
        dummy_config = BacktestConfig(test_limit_partitions=OPTUNA_TEST_LIMIT)
        dummy_simulator = BacktestSimulator(dummy_config)
        data = dummy_simulator.preload_data()
        with open(S7_BACKTEST_CACHE, "wb") as f:
            pickle.dump(data, f)
        logging.info(f"キャッシュ保存: {S7_BACKTEST_CACHE}")
        return data

    global_preloaded_data = load_or_generate_cache()
    logging.info("Data preload complete! Starting Optuna...")

    # =========================================================
    # Optuna最適化セッション
    # =========================================================
    db_path = str(S7_BACKTEST_OPTUNA_RESULTS / "optuna_baseline_atr_v2.db")

    # 格子探索: 2 × 3 = 6 trials
    search_space = {
        "m2_th":    [0.70, 0.80],
        "m2_delta": [0.30, 0.50, 0.80],
    }
    study_name = "baseline_atr_search_v2"

    study = optuna.create_study(
        study_name=study_name,
        storage=f"sqlite:///{db_path}",
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.GridSampler(search_space),
    )

    n_trials = 6  # 2 × 3

    logging.info(f"Study: {study_name} | n_trials: {n_trials} (2×3 grid)")
    logging.info("固定: min_atr_threshold=0.90, min_baseline_atr=0.0, risk=2%")
    study.optimize(objective, n_trials=n_trials, callbacks=[save_csv_callback])

    # =========================================================
    # 最終ランキング出力
    # =========================================================
    print("\n\n" + "="*85)
    print("  📊 FINAL RESULTS — M2_th × Delta Optimization")
    print("="*85)
    export_rankings(study)

    best = study.best_trial
    print(f"\n  Best Trial : #{best.number:03d}")
    print(f"  Best Score : {study.best_value:.4f}")
    for k, v in best.params.items():
        print(f"    {k}: {v}")
    print(f"  Best PF    : {best.user_attrs.get('Profit_Factor', 'N/A'):.2f}")
    print(f"  Best Trades: {best.user_attrs.get('Total_Trades', 'N/A')}")
    print("="*85)
