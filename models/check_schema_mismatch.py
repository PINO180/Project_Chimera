"""
check_schema_mismatch.py
========================
再実行前の事前診断スクリプト

【確認内容】
1. m2_long_features.txt / m2_short_features.txt の各カラムが
   meta_labeled_oof_long / meta_labeled_oof_short に実際に存在するか
2. 同様に m1_*_features.txt が weighted_dataset に存在するか
3. tmp_m2_oof_predictions_long/short の残骸が残っていないか
4. 較正ファイル・モデルファイルの退避漏れがないか（タイムスタンプ確認）
5. Cx版で追加された退避漏れの可能性があるファイル一覧

使い方:
    python check_schema_mismatch.py

出力:
    コンソールに診断結果を表示 + check_result.json に保存
"""

import sys
import json
import datetime
from pathlib import Path
from typing import List, Dict

import polars as pl

# ── blueprint のパスをそのまま流用 ──────────────────────────────
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

try:
    from blueprint import (
        S3_SELECTED_FEATURES_DIR,
        S7_MODELS,
        S7_META_LABELED_OOF_LONG,
        S7_META_LABELED_OOF_SHORT,
        S6_WEIGHTED_DATASET,
        S7_M2_OOF_PREDICTIONS_TMP_LONG,
        S7_M2_OOF_PREDICTIONS_TMP_SHORT,
        S7_M1_CALIBRATED_LONG,
        S7_M1_CALIBRATED_SHORT,
        S7_M2_CALIBRATED_LONG,
        S7_M2_CALIBRATED_SHORT,
        S7_M1_MODEL_LONG_PKL,
        S7_M1_MODEL_SHORT_PKL,
        S7_M2_MODEL_LONG_PKL,
        S7_M2_MODEL_SHORT_PKL,
        S7_M2_OOF_PREDICTIONS_LONG,
        S7_M2_OOF_PREDICTIONS_SHORT,
    )
    print("[OK] blueprint.py を読み込みました\n")
except ImportError as e:
    print(f"[ERROR] blueprint.py のインポートに失敗: {e}")
    print("  → このスクリプトを /workspace/models/ に置いて実行してください")
    sys.exit(1)

SEPARATOR = "=" * 70
results: Dict = {}

# ────────────────────────────────────────────────────────────────
# ユーティリティ
# ────────────────────────────────────────────────────────────────

def load_feature_list(path: Path) -> List[str]:
    """テキストファイルから特徴量リストを読み込む（Cx版 _load_features と同じ除外ロジック）"""
    exclude_exact = {
        "timestamp", "t1", "label", "label_long", "label_short",
        "uniqueness", "uniqueness_long", "uniqueness_short",
        "payoff_ratio", "payoff_ratio_long", "payoff_ratio_short",
        "pt_multiplier", "sl_multiplier", "direction", "exit_type",
        "first_ex_reason_int", "atr_value", "calculated_body_ratio",
        "fallback_vol", "open", "high", "low", "close",
        "meta_label", "m1_pred_proba", "is_trigger",
    }
    with open(path, "r") as f:
        raw = [line.strip() for line in f if line.strip()]

    features = ["timeframe"]
    for col in raw:
        if col in exclude_exact or col == "timeframe":
            continue
        if col.startswith("is_trigger_on"):
            continue
        features.append(col)
    return features


def get_parquet_columns(hive_dir: Path, n_sample: int = 3) -> List[str]:
    """Hiveパーティションから先頭 n_sample ファイルのカラム一覧を取得"""
    files = sorted(hive_dir.glob("year=*/month=*/day=*/*.parquet"))[:n_sample]
    if not files:
        return []
    # 最初の 1 ファイルだけスキャンすれば十分（スキーマは全パーティション共通のはず）
    df = pl.scan_parquet(str(files[0])).collect()
    return df.columns


def fmt_ts(ts: float) -> str:
    return datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


# ────────────────────────────────────────────────────────────────
# 診断 1: tmp 残骸チェック
# ────────────────────────────────────────────────────────────────
print(SEPARATOR)
print("【診断 1】tmp ディレクトリの残骸チェック")
print(SEPARATOR)

tmp_check = {}
for label, tmp_path in [
    ("LONG ", S7_M2_OOF_PREDICTIONS_TMP_LONG),
    ("SHORT", S7_M2_OOF_PREDICTIONS_TMP_SHORT),
]:
    exists = tmp_path.exists()
    if exists:
        children = list(tmp_path.iterdir())
        n_parquet = len(list(tmp_path.glob("**/*.parquet")))
        status = f"⚠️  存在する（子要素: {len(children)}件, parquetファイル: {n_parquet}件）"
        tmp_check[label] = {"exists": True, "children": len(children), "parquet_files": n_parquet}
    else:
        status = "✅ 存在しない（正常）"
        tmp_check[label] = {"exists": False}
    print(f"  {label}: {tmp_path.name}  →  {status}")

results["tmp_check"] = tmp_check

# ────────────────────────────────────────────────────────────────
# 診断 2: 較正・モデルファイルの退避漏れチェック（タイムスタンプ付き）
# ────────────────────────────────────────────────────────────────
print()
print(SEPARATOR)
print("【診断 2】stratum_7_models 内の既存ファイル一覧（退避漏れ確認）")
print(SEPARATOR)
print("  ※ 再実行前にこれらが残っていると SKIPPING が発動して結果が汚染されます")
print()

watch_files = {
    "m1_calibrated_long":  S7_M1_CALIBRATED_LONG,
    "m1_calibrated_short": S7_M1_CALIBRATED_SHORT,
    "m2_calibrated_long":  S7_M2_CALIBRATED_LONG,
    "m2_calibrated_short": S7_M2_CALIBRATED_SHORT,
    "m1_model_long":       S7_M1_MODEL_LONG_PKL,
    "m1_model_short":      S7_M1_MODEL_SHORT_PKL,
    "m2_model_long":       S7_M2_MODEL_LONG_PKL,
    "m2_model_short":      S7_M2_MODEL_SHORT_PKL,
    "m2_oof_long":         S7_M2_OOF_PREDICTIONS_LONG,
    "m2_oof_short":        S7_M2_OOF_PREDICTIONS_SHORT,
}

file_status = {}
for label, path in watch_files.items():
    if path.exists():
        stat = path.stat()
        ts = fmt_ts(stat.st_mtime)
        size_kb = stat.st_size // 1024
        print(f"  ⚠️  {path.name:<40} 更新: {ts}  ({size_kb:,} KB)")
        file_status[label] = {"exists": True, "mtime": ts, "size_kb": size_kb}
    else:
        print(f"  ✅  {path.name:<40} 存在しない")
        file_status[label] = {"exists": False}

results["file_status"] = file_status

# ────────────────────────────────────────────────────────────────
# 診断 3: スキーマ一致チェック（メイン）
# ────────────────────────────────────────────────────────────────
print()
print(SEPARATOR)
print("【診断 3】特徴量リスト vs 実データ スキーマ一致チェック")
print(SEPARATOR)

schema_results = {}

checks = [
    # (ラベル, 特徴量ファイル, データディレクトリ, M2かどうか)
    ("M1 LONG  feat vs weighted_dataset",
     S3_SELECTED_FEATURES_DIR / "m1_long_features.txt",
     S6_WEIGHTED_DATASET, False),
    ("M1 SHORT feat vs weighted_dataset",
     S3_SELECTED_FEATURES_DIR / "m1_short_features.txt",
     S6_WEIGHTED_DATASET, False),
    ("M2 LONG  feat vs meta_labeled_oof_long",
     S3_SELECTED_FEATURES_DIR / "m2_long_features.txt",
     S7_META_LABELED_OOF_LONG, True),
    ("M2 SHORT feat vs meta_labeled_oof_short",
     S3_SELECTED_FEATURES_DIR / "m2_short_features.txt",
     S7_META_LABELED_OOF_SHORT, True),
]

for label, feat_path, data_dir, is_m2 in checks:
    print()
    print(f"  ▶ {label}")

    if not feat_path.exists():
        print(f"    [ERROR] 特徴量ファイルが見つかりません: {feat_path}")
        schema_results[label] = {"error": "feature file not found"}
        continue

    if not data_dir.exists():
        print(f"    [ERROR] データディレクトリが見つかりません: {data_dir}")
        schema_results[label] = {"error": "data dir not found"}
        continue

    # 特徴量リスト取得
    features = load_feature_list(feat_path)
    if is_m2 and "m1_pred_proba" not in features:
        features.append("m1_pred_proba")

    # データのカラム取得
    data_cols = get_parquet_columns(data_dir)
    if not data_cols:
        print(f"    [ERROR] parquet ファイルが見つかりません: {data_dir}")
        schema_results[label] = {"error": "no parquet files"}
        continue

    data_col_set = set(data_cols)

    # 欠落カラム（特徴量リストにあるがデータにない）
    missing = [f for f in features if f not in data_col_set]
    # 余剰カラム（データにあるが特徴量リストにない、参考情報）
    extra_in_data = [c for c in data_cols if c not in set(features)]

    n_feat = len(features)
    n_missing = len(missing)

    if n_missing == 0:
        print(f"    ✅ 全 {n_feat} 特徴量が一致しています")
    else:
        print(f"    ❌ {n_feat} 特徴量中 {n_missing} 件が【データに存在しない】")
        print(f"       → これらの欠落カラムが原因で全パーティションが continue されます")
        print(f"       欠落カラム一覧:")
        for col in missing:
            print(f"         - {col}")

    print(f"    データ側の総カラム数: {len(data_cols)}")
    print(f"    データにのみ存在するカラム数（参考）: {len(extra_in_data)}")

    schema_results[label] = {
        "n_features": n_feat,
        "n_missing": n_missing,
        "missing_columns": missing,
        "data_total_columns": len(data_cols),
        "extra_in_data_count": len(extra_in_data),
    }

results["schema_check"] = schema_results

# ────────────────────────────────────────────────────────────────
# 診断 4: m2_oof_predictions_long.parquet の中身確認
# ────────────────────────────────────────────────────────────────
print()
print(SEPARATOR)
print("【診断 4】現在の m2_oof_predictions_long.parquet の中身確認")
print(SEPARATOR)

oof_check = {}
for label, oof_path in [
    ("LONG ", S7_M2_OOF_PREDICTIONS_LONG),
    ("SHORT", S7_M2_OOF_PREDICTIONS_SHORT),
]:
    if not oof_path.exists():
        print(f"  {label}: 存在しない")
        oof_check[label] = {"exists": False}
        continue

    try:
        df = pl.read_parquet(oof_path)
        n_rows = len(df)
        cols = df.columns
        has_true_label = "true_label" in cols
        n_null_label = df["true_label"].null_count() if has_true_label else -1
        print(f"  {label}: {n_rows:,} 行  カラム: {cols}")
        print(f"         true_label 列: {'あり' if has_true_label else '【なし】'}"
              + (f"  null数: {n_null_label}" if has_true_label else ""))
        oof_check[label] = {
            "exists": True, "rows": n_rows, "columns": cols,
            "has_true_label": has_true_label,
            "null_true_label": n_null_label,
        }
    except Exception as e:
        print(f"  {label}: 読み込みエラー → {e}")
        oof_check[label] = {"exists": True, "error": str(e)}

results["oof_check"] = oof_check

# ────────────────────────────────────────────────────────────────
# 診断 5: 再実行前チェックリスト サマリ
# ────────────────────────────────────────────────────────────────
print()
print(SEPARATOR)
print("【診断 5】再実行前チェックリスト")
print(SEPARATOR)

ok_count = 0
ng_count = 0

def chk(cond: bool, msg_ok: str, msg_ng: str):
    global ok_count, ng_count
    if cond:
        print(f"  ✅ {msg_ok}")
        ok_count += 1
    else:
        print(f"  ❌ {msg_ng}")
        ng_count += 1

# tmp残骸なし
for label, info in tmp_check.items():
    chk(not info["exists"],
        f"tmp_{label.strip().lower()} は存在しない（正常）",
        f"tmp_{label.strip().lower()} が残っている → 削除してから再実行すること")

# 較正・モデルファイルが存在しない（クリーンな状態）
for label, info in file_status.items():
    if "calibrated" in label:
        chk(not info["exists"],
            f"{label} の較正ファイルは存在しない（正常）",
            f"{label} の較正ファイルが残存 → SKIPPINGが発動するため退避・削除すること")

# スキーマ一致
for label, info in schema_results.items():
    if "error" in info:
        chk(False, "", f"{label}: ファイル/ディレクトリが見つからず確認不可")
    else:
        chk(info["n_missing"] == 0,
            f"{label}: スキーマ一致",
            f"{label}: {info['n_missing']} カラム欠落 → 再実行しても同じ結果になります")

print()
print(f"  合計: ✅ {ok_count} 件OK  /  ❌ {ng_count} 件NG")

# ────────────────────────────────────────────────────────────────
# JSON保存
# ────────────────────────────────────────────────────────────────
out_path = Path(__file__).parent / "check_result.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2, default=str)

print()
print(SEPARATOR)
print(f"診断結果を保存しました → {out_path}")
print(SEPARATOR)
