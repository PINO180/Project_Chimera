# /workspace/backtest_simulator/backtest_simulator_run_optuna.py
import sys
import shutil  # ★追加: ファイルコピー用
from pathlib import Path
import logging
import optuna
import pandas as pd

# --- プロジェクトのルートディレクトリをPythonの検索パスに追加 ---
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from backtest_simulator_cimera_margin_level_optuna import (
    BacktestConfig,
    BacktestSimulator,
)  # noqa: E402

# --- Optuna用ロギング設定 ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - [OPTUNA] %(levelname)s - %(message)s"
)

# ==========================================
# オプティマイザーの全体設定
# ==========================================
OPTUNA_TEST_LIMIT = 0  # 探索にかけるパーティション数（0=全期間）
MAX_ALLOWED_DRAWDOWN = (
    30.0  # 許容する最大ドローダウン（%）。これを超えると問答無用で不採用
)
# ==========================================

# ★ オンメモリ展開したデータを保持するためのグローバル変数
global_preloaded_data = None


def objective(trial: optuna.Trial):
    """
    指定されたリストの中からのみパラメータを選択し、最適化を行う目的関数。
    """
    # =========================================================
    # 1. Optunaに探索させるパラメータ（Categorical指定）
    # =========================================================
    fixed_risk_pct = trial.suggest_categorical(
        "fixed_risk_pct",
        [
            # 0.01,
            # 0.02,
            # 0.03,
            0.05,
            # 0.10,
            # 0.15,
            # 0.20,
        ],
    )
    m2_th = trial.suggest_categorical(
        "m2_th",
        [
            # 0.00,
            # 0.10,
            # 0.20,
            0.30,
            # 0.40,
            # 0.50,
            # 0.55,
            # 0.60,
            # 0.70,
            # 0.80,
            # 0.90,
            # 0.95,
        ],
    )

    # ▼▼▼ 追加: 差分(Delta)の探索パラメータ ▼▼▼
    m2_delta = trial.suggest_categorical(
        "m2_delta",
        [
            0.00,  # 差分ゼロ（Trial 7 の完全同意モードを含む）
            # 0.10,
            # 0.20,
            0.30,
            # 0.40,
            0.50,  # 圧倒的優勢モード
            # 0.60,
            # 0.70,
            0.80,
        ],
    )
    # ▲▲▲ ここまで追加 ▲▲▲

    max_pos = trial.suggest_categorical(
        "max_positions",
        [
            # 1, 2,5, 10, 20, 30,50,
            100,
        ],
    )
    max_cons_sl = trial.suggest_categorical(
        "max_consecutive_sl",
        [
            # 1,
            2,
            # 3,
        ],
    )
    cooldown = trial.suggest_categorical(
        "cooldown_minutes_after_sl",
        [
            # 5, 10,15,
            30,
            # 60,120,
        ],
    )

    # =========================================================
    # 2. 探索パラメータ ＆ 確定済み聖杯パラメータのセット
    # =========================================================
    config = BacktestConfig(
        use_fixed_risk=True,
        fixed_risk_percent=fixed_risk_pct,
        m2_proba_threshold=m2_th,
        m2_delta_threshold=m2_delta,  # ★追加
        max_positions=max_pos,
        max_consecutive_sl=max_cons_sl,
        cooldown_minutes_after_sl=cooldown,
        test_limit_partitions=OPTUNA_TEST_LIMIT,
        oof_mode=True,
        min_atr_threshold=2.0,
        sl_multiplier_long=5.0,
        pt_multiplier_long=1.0,
        sl_multiplier_short=5.0,
        pt_multiplier_short=1.0,
        td_minutes_long=60.0,
        td_minutes_short=60.0,
        margin_call_percent=100.0,
        stop_out_percent=20.0,
    )

    logging.info(f"--- Starting Trial {trial.number} ---")
    logging.info(
        f"Params: Risk={fixed_risk_pct * 100:.0f}%, M2_th={m2_th}, Delta={m2_delta}, MaxSL={max_cons_sl}"
    )

    logger = logging.getLogger()
    original_level = logger.level
    logger.setLevel(logging.WARNING)  # ループ中は余計なログを消す

    try:
        simulator = BacktestSimulator(config)
        report_data = simulator.run(preloaded_data=global_preloaded_data)
    except Exception as e:
        logger.setLevel(original_level)
        logging.error(f"Trial {trial.number} failed with exception: {e}")
        raise optuna.exceptions.TrialPruned()

    logger.setLevel(original_level)

    # =========================================================
    # ★追加: 出力された4つのファイルを専用フォルダに退避させる
    # =========================================================
    # シミュレーターがファイルを出力する元の場所
    src_dir = Path("/workspace/data/XAUUSD/stratum_7_models/1A_2B")

    # フォルダ名に m2_th と m2_delta も含めて分かりやすくする
    folder_name = f"Trial_{trial.number}_Th{m2_th}_Delta{m2_delta}_Risk{fixed_risk_pct * 100:.0f}_SL{max_cons_sl}"
    dst_dir = Path(f"/workspace/backtest_simulator/optuna_results/{folder_name}")
    dst_dir.mkdir(parents=True, exist_ok=True)

    files_to_copy = [
        "final_backtest_report_v5.json",
        "final_backtest_report_v5.txt",
        "equity_curve_v5.png",
        "detailed_trade_log_v5.csv",
    ]

    for fname in files_to_copy:
        src_file = src_dir / fname
        if src_file.exists():
            shutil.copy2(src_file, dst_dir / fname)
    # =========================================================

    # =========================================================
    # 4. 評価指標の抽出 ＆ CSV保存用の属性セット
    # =========================================================
    max_dd = report_data.get("max_drawdown_pct", 0.0)
    sharpe = report_data.get("sharpe_ratio_annual", 0.0)
    profit_factor = report_data.get("profit_factor", 0.0)
    trades = report_data.get("total_trades", 0)
    min_margin = report_data.get("min_margin_level_pct", 9999.0)
    stop_outs = report_data.get("stop_out_count", 0)

    trial.set_user_attr(
        "Total_Net_Profit",
        report_data.get("final_capital", 0.0) - report_data.get("initial_capital", 0.0),
    )
    trial.set_user_attr("Max_DD_Pct", max_dd)
    trial.set_user_attr("Sharpe_Ratio", sharpe)
    trial.set_user_attr("Profit_Factor", profit_factor)
    trial.set_user_attr("Total_Trades", trades)
    trial.set_user_attr(
        "Min_Margin_Pct", float(min_margin) if min_margin != 9999.0 else None
    )

    # =========================================================
    # 5. ペナルティ判定
    # =========================================================
    if trades < 50:
        raise optuna.exceptions.TrialPruned()

    if stop_outs > 0:
        return -9999.0

    if max_dd > MAX_ALLOWED_DRAWDOWN:
        return -9999.0

    if min_margin < 150.0:
        penalty = (150.0 - min_margin) * 10
        sharpe -= penalty

    score = sharpe + (profit_factor * 0.1)
    return float(score)


def export_rankings_to_csv(study: optuna.Study):
    """全トライアル終了後、結果をPandasで集計してCSVにランキング出力する関数"""
    logging.info("Exporting rankings to CSV...")

    trials_data = []
    for t in study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue

        row = {"Trial_ID": t.number, "Score": t.value, **t.params, **t.user_attrs}
        trials_data.append(row)

    if not trials_data:
        logging.warning("No complete trials to export.")
        return

    df = pd.DataFrame(trials_data)

    # ▼▼▼ この1行を足すだけ ▼▼▼
    df = df[df["Score"] != -9999.0]

    # --- シャープレシオ 上位100 ---
    df_sharpe = df.sort_values(by="Sharpe_Ratio", ascending=False).head(100)
    out_sharpe = Path("/workspace/backtest_simulator/optuna_ranking_top100_sharpe.csv")
    df_sharpe.to_csv(out_sharpe, index=False)
    logging.info(f"Saved Sharpe Ratio top 100 ranking to: {out_sharpe}")

    # --- Total Net Profit 上位100 ---
    df_np = df.sort_values(by="Total_Net_Profit", ascending=False).head(100)
    out_np = Path("/workspace/backtest_simulator/optuna_ranking_top100_net_profit.csv")
    df_np.to_csv(out_np, index=False)
    logging.info(f"Saved Total Net Profit top 100 ranking to: {out_np}")


# ▼▼▼ ここから追加・修正 ▼▼▼
def save_csv_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial):
    """1回のTrialが終わるたびにランキングCSVをリアルタイム更新する"""
    export_rankings_to_csv(study)


if __name__ == "__main__":
    # =========================================================
    # ★ 超速化のキモ：Optuna開始前に、データを1回だけメモリに全ロードする
    # =========================================================56
    logging.info("Initializing pre-load sequence for Ultra-Fast Optuna...")
    dummy_config = BacktestConfig(test_limit_partitions=OPTUNA_TEST_LIMIT)
    dummy_simulator = BacktestSimulator(dummy_config)

    # global global_preloaded_data
    global_preloaded_data = dummy_simulator.preload_data()
    logging.info("Data preload complete! Starting rapid Optuna trials...")

    # =========================================================
    # Optuna 最適化セッションの開始
    # =========================================================
    db_path = "sqlite:///optuna_v5_tuning.db"

    # ▼修正: 新しいパラメータ(Delta)専用のテーブル名にする
    study_name = "v5_delta_filter_optimization"

    study = optuna.create_study(
        study_name=study_name,
        storage=db_path,
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    logging.info(f"Starting Optuna optimization. Study: {study_name}")

    n_trials = 5

    # ★ 修正ポイント：callbacks=[save_csv_callback] を追加！
    # これにより、1周（約2分）終わるごとに自動でCSVが更新されます
    study.optimize(objective, n_trials=n_trials, callbacks=[save_csv_callback])

    print("\n" + "=" * 50)
    print(" 🏆 Optimization Finished!")
    print("=" * 50)
    print(f" Best Trial: {study.best_trial.number}")
    print(f" Best Score: {study.best_value:.4f}")
    print(" Best Params: ")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")
    print("=" * 50)
