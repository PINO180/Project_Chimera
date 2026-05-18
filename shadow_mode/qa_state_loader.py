#!/usr/bin/env python3
"""
[Phase 9d 発見 #66 Phase D-3] QAState seed artifact loader (shadow_mode 共通)

main.py L640-694 と同じ load 経路を shadow_mode 各スクリプトから呼び出せる
ようにした共通ヘルパー。これにより本番側起動経路と shadow_mode 検証経路で
QAState の初期化状態が完全一致し、Layer 1 Shadow Mode の比較結果が本番
ライブと数値同一になることが保証される。

責務:
  1. S3_QA_STATES_DIR / qa_state_e1{a..f}.pkl を 6 engine 分 load
  2. 1 つでも欠落していた engine は ERROR ログ + 旧挙動 fallback
     (silent fallback ではなく明示的に Train-Serve Skew が残ることを通知)
  3. ShadowEngine(qa_state_artifacts=...) に渡す形式の dict を返す
  4. キャッシュキー無効化用に pickle ファイル群の (path, mtime) を返す
     ヘルパーも同梱

main.py との SSoT 同期:
  load ロジックは main.py L640-694 と byte-identical な挙動を保つ。
  変更がある場合は両方を同時更新する (互いの divergence は Train-Serve
  Skew の原因になる)。

  ※ 形式: dict {engine_id: {(tf, feat): {ewm_mean, ewm_var, ewm_n}}}
  ※ 1 件も load 成功しなかった場合は None を返し、ShadowEngine 側で
     旧挙動 fallback (warmup loop で seed) に遷移する。
"""
from __future__ import annotations

import logging
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Allow imports from /workspace
_WORKSPACE = Path(__file__).resolve().parents[1]
if str(_WORKSPACE) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE))

import blueprint as config  # noqa: E402

logger = logging.getLogger("shadow_mode.qa_state_loader")

# 学習側 engine_1_A 〜 1_F が出力する 6 pickle の engine_id 一覧。
# 順序は main.py L659 と一致させる (ログ出力順の互換性のため)。
_ENGINE_IDS: Tuple[str, ...] = ("e1a", "e1b", "e1c", "e1d", "e1e", "e1f")


def artifact_pickle_paths(base_dir: Optional[Path] = None) -> List[Path]:
    """Phase D-3 で扱う 6 pickle の絶対パスを返す (存在しなくても列挙)。

    cache key 生成で mtime を含めるために使う。これにより再学習で
    pickle が再生成されたら shadow_mode 側 warmup cache が自動無効化される。

    [Phase D-3 修正方針 R] base_dir:
        None (default) → blueprint.S3_QA_STATES_DIR (本番ライブ用 artifact dir)
        Path 指定      → そのパス配下の qa_state_{engine_id}.pkl を見る
                          (Layer 1 検証用 cutoff_YYYYMMDD/ subdir 等)
    """
    root = Path(base_dir) if base_dir is not None else config.S3_QA_STATES_DIR
    return [root / f"qa_state_{eid}.pkl" for eid in _ENGINE_IDS]


def load_qa_state_artifacts(
    base_dir: Optional[Path] = None,
) -> Optional[Dict[str, Dict]]:
    """[Phase D-3] 学習側 QAState seed artifact を 6 engine 分 load する。

    main.py L640-694 と同じロジック (silent fallback を避けるため欠落は
    ERROR ログ、全欠落のみ旧挙動 fallback で None を返す)。

    Args:
        base_dir: artifact dir をオーバーライド。None = 本番ライブ用 dir
                  (= blueprint.S3_QA_STATES_DIR)。Layer 1 検証では学習側
                  --cut-off-date 指定で生成した別 dir を指定する。

    Returns:
        - dict {engine_id: {(tf, feat): {ewm_mean, ewm_var, ewm_n}}}
          (1 件以上 load 成功した場合、欠落 engine は dict に含めない)
        - None: 1 件も load 成功しなかった場合 (旧挙動 fallback 用シグナル)

    Raises:
        例外は raise しない。load 失敗 / 不在 は ERROR ログのみ。
    """
    root = Path(base_dir) if base_dir is not None else config.S3_QA_STATES_DIR
    if base_dir is not None:
        logger.info(f"[Phase D-3] artifact dir override: {root}")
    artifacts: Dict[str, Dict] = {}
    all_loaded = True
    for engine_id in _ENGINE_IDS:
        path = root / f"qa_state_{engine_id}.pkl"
        if path.exists():
            try:
                with path.open("rb") as f:
                    artifacts[engine_id] = pickle.load(f)
                logger.info(
                    f"[Phase D-3] QAState artifact load OK: {path.name} "
                    f"({len(artifacts[engine_id])} entries)"
                )
            except Exception as e:
                # silent fallback を避ける: load 失敗は ERROR ログ + 当該
                # engine だけ旧挙動 fallback (他 engine は正常 load 済)。
                logger.error(
                    f"[Phase D-3] QAState artifact load 失敗: {path}: {e}. "
                    f"engine={engine_id} は旧挙動で fallback。"
                )
                all_loaded = False
        else:
            logger.error(
                f"[Phase D-3] QAState artifact 不在: {path}. "
                f"engine={engine_id} は旧挙動で fallback "
                f"(Train-Serve Skew が残存します)。"
            )
            all_loaded = False

    if not all_loaded:
        logger.warning(
            "[Phase D-3] QAState artifact が一部欠落しています。"
            "初回 deploy 直後で artifact 未生成の場合は次回学習後に "
            "解消されますが、それ以外の場合は S3_QA_STATES_DIR を確認。"
        )

    return artifacts if artifacts else None


def summarize_artifacts(
    artifacts: Optional[Dict[str, Dict]],
) -> str:
    """artifact dict のサマリーログ用文字列を生成する (デバッグ補助)。"""
    if artifacts is None:
        return "(none — 旧挙動 fallback)"
    engines = sorted(artifacts.keys())
    total = sum(len(art) for art in artifacts.values())
    return f"engines={engines}, total_entries={total}"
