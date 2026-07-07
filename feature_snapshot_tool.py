# feature_snapshot_tool.py
# =====================================================================
# ツール①: 特徴量スナップショット出力ツール
#
# 【目的】
#   シグナル判定の瞬間(HOLD含む全 M3 close)の全特徴量値を記録する。
#   本番が生成している特徴量の値が学習時の分布と乖離していないかを
#   確認するための診断ツール。
#
# 【出力形式 — 2 系統を併存 (Phase 11.32.x: (R) 案)】
#   (1) 縦持ち (従来通り、 1 判定 1 ファイル):
#         /workspace/data/diagnostics/feature_snapshots/
#           snapshot_YYYYMMDD_HHMMSS_L{m2l:.3f}_S{m2s:.3f}.csv
#       列: feature_name, value  (メタ行 + 全特徴量を縦に並べる)
#       → diagnose_volume_skew.py 等 既存診断 6 本がこの形式を入力にするため温存。
#
#   (2) 横持ち (新規、 単一 CSV へ 1 判定 1 行 追記):
#         /workspace/data/diagnostics/feature_snapshots_wide.csv
#       列: Timestamp, _m1_long, _m1_short, _m2_long, _m2_short, _atr_ratio,
#           <feature_dict の全 key を sorted で展開>
#       → compare_3way_ols.py が --production に直接食える wide 形式。
#         triggered_features_log.csv (発注時のみ) と違い HOLD 含む全判定を記録するため、
#         発注を待たずに比較サンプルが毎 M3 close 1 行ずつ溜まる。
#
# 【使い方】 main.py のシグナル処理部分 (M2 フィルタ判定の前) で:
#   from feature_snapshot_tool import save_feature_snapshot
#   save_feature_snapshot(signal.feature_dict, p_long_m1_raw, p_short_m1_raw,
#                         p_long_m2_raw, p_short_m2_raw, atr_ratio)
#   ※ 呼び出し側の引数は従来と完全に同一。本ツール単体の変更で横持ちを追加。
# =====================================================================

import csv
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

OUTPUT_DIR = Path("/workspace/data/diagnostics/feature_snapshots")
WIDE_CSV_PATH = Path("/workspace/data/diagnostics/feature_snapshots_wide.csv")
logger = logging.getLogger("♾️Chimera♾️.SNAP")

# 横持ち CSV のメタ列 (feature_dict の特徴量列より前に置く固定列)
_WIDE_META_COLS = [
    "Timestamp",
    "_m1_long",
    "_m1_short",
    "_m2_long",
    "_m2_short",
    "_atr_ratio",
]


def _append_wide_row(
    feature_dict: Dict[str, float],
    now_iso: str,
    p_m1_long: float,
    p_m1_short: float,
    p_m2_long: float,
    p_m2_short: float,
    atr_ratio: float,
) -> None:
    """横持ち単一 CSV へ 1 判定 1 行を追記する。

    ヘッダーは初回のみ書き込む。 feature_dict の key 集合は毎判定同一である
    前提だが、 既存ヘッダーと現在の key 集合が食い違った場合は警告ログを出す
    (triggered_features_log.csv L1434 のヘッダー凍結問題と同種の事故を検出する
    ための安全弁)。
    """
    feature_keys = sorted(feature_dict.keys())
    expected_header = _WIDE_META_COLS + feature_keys

    file_exists = WIDE_CSV_PATH.exists()

    # --- 既存ヘッダーとの整合チェック (安全弁) ---
    existing_header = None
    if file_exists:
        try:
            with open(WIDE_CSV_PATH, "r", newline="", encoding="utf-8") as rf:
                reader = csv.reader(rf)
                existing_header = next(reader, None)
        except Exception:
            existing_header = None

    if existing_header is not None and existing_header != expected_header:
        existing_feat = set(existing_header) - set(_WIDE_META_COLS)
        current_feat = set(feature_keys)
        missing = sorted(current_feat - existing_feat)  # 既存ヘッダーに無い新 key
        dropped = sorted(existing_feat - current_feat)  # 今回 dict に無い旧 key
        logger.warning(
            "📸 [WideSnapshot] feature key set differs from header. "
            f"new={len(missing)}, dropped={len(dropped)}. "
            "Keeping existing header order (new keys are not recorded). "
            f"e.g. new={missing[:3]}, dropped={dropped[:3]}"
        )

    # 記録に使う列順: 既存ヘッダーがあればそれに従う (凍結維持)、 無ければ expected
    write_header = existing_header if existing_header is not None else expected_header
    write_feature_keys = (
        [c for c in write_header if c not in _WIDE_META_COLS]
        if existing_header is not None
        else feature_keys
    )

    meta_val = {
        "Timestamp": now_iso,
        "_m1_long": round(p_m1_long, 6),
        "_m1_short": round(p_m1_short, 6),
        "_m2_long": round(p_m2_long, 6),
        "_m2_short": round(p_m2_short, 6),
        "_atr_ratio": round(atr_ratio, 6),
    }

    with open(WIDE_CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(write_header)
        row = [meta_val[c] for c in _WIDE_META_COLS]
        # key 不在は空欄 (= 計算結果なし) として記録
        row += [feature_dict[k] if k in feature_dict else "" for k in write_feature_keys]
        writer.writerow(row)


def save_feature_snapshot(
    feature_dict: Dict[str, float],
    p_m1_long: float,
    p_m1_short: float,
    p_m2_long: float,
    p_m2_short: float,
    atr_ratio: float,
    max_snapshots: int = 500,
) -> None:
    """
    シグナル判定時の全特徴量値を縦持ち(1 判定 1 ファイル) + 横持ち(単一 CSV 追記)
    の両形式で保存する。

    Args:
        feature_dict: signal.feature_dict（全特徴量の辞書）
        p_m1_long:    M1 Long予測値
        p_m1_short:   M1 Short予測値
        p_m2_long:    M2 Long予測値
        p_m2_short:   M2 Short予測値
        atr_ratio:    ATR Ratio
        max_snapshots: 縦持ちで保持する最大ファイル数（古いものから削除）
    """
    now = datetime.now(timezone.utc)
    now_iso = now.isoformat()

    # ============================================================
    # (1) 縦持ち (従来通り) — 既存診断 6 本のため温存
    # ============================================================
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        fname = (
            f"snapshot_{now.strftime('%Y%m%d_%H%M%S')}"
            f"_L{p_m2_long:.3f}_S{p_m2_short:.3f}.csv"
        )
        fpath = OUTPUT_DIR / fname

        with open(fpath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["feature_name", "value"])
            writer.writerow(["_timestamp_utc", now_iso])
            writer.writerow(["_m1_long", round(p_m1_long, 6)])
            writer.writerow(["_m1_short", round(p_m1_short, 6)])
            writer.writerow(["_m2_long", round(p_m2_long, 6)])
            writer.writerow(["_m2_short", round(p_m2_short, 6)])
            writer.writerow(["_atr_ratio", round(atr_ratio, 6)])
            writer.writerow(["---", "---"])
            for name, val in sorted(feature_dict.items()):
                writer.writerow([name, val])

        logger.info(
            f"🏭 [Snapshot] saved (long): {fname} ({len(feature_dict)} features)"
        )

        # 古いファイルを削除（max_snapshots超過分）
        existing = sorted(OUTPUT_DIR.glob("snapshot_*.csv"))
        if len(existing) > max_snapshots:
            for old_file in existing[: len(existing) - max_snapshots]:
                old_file.unlink()
                logger.debug(f"old snapshot deleted: {old_file.name}")

    except Exception as e:
        logger.error(f"long-format snapshot save failed: {e}", exc_info=True)

    # ============================================================
    # (2) 横持ち (新規) — compare_3way_ols.py 用の単一 wide CSV
    # ============================================================
    try:
        WIDE_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
        _append_wide_row(
            feature_dict=feature_dict,
            now_iso=now_iso,
            p_m1_long=p_m1_long,
            p_m1_short=p_m1_short,
            p_m2_long=p_m2_long,
            p_m2_short=p_m2_short,
            atr_ratio=atr_ratio,
        )
    except Exception as e:
        logger.error(f"wide-format snapshot save failed: {e}", exc_info=True)
