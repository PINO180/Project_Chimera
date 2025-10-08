"""
merge_validation_results.py
第一防衛線 - 検証結果統合スクリプト

3つの独立した検証テストの結果を統合し、
最終的な安定特徴量リストを生成
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import blueprint as config
import logging
from pathlib import Path
import json
from datetime import datetime
from typing import List, Set

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import joblib
import pandas as pd


class ProcessingConfig:
    """設定管理クラス"""
    def __init__(self):
        self.artifacts_dir = config.S3_ARTIFACTS
        self.ks_results_path = self.artifacts_dir / "ks_test_results.json"
        self.pi_results_path = self.artifacts_dir / "pi_test_results.json"
        self.adversarial_results_path = self.artifacts_dir / "adversarial_test_results.json"
        self.adversarial_scores_path = self.artifacts_dir / "adversarial_scores.joblib"
        
        # 出力パス
        self.stable_list_joblib = config.S3_STABLE_FEATURE_LIST
        self.stable_list_csv = self.artifacts_dir / "stable_feature_list.csv"
        self.stable_list_json = self.artifacts_dir / "stable_feature_list.json"
        self.adversarial_scores_final = config.S3_ADVERSARIAL_SCORES


class ValidationResultMerger:
    """検証結果統合クラス"""
    
    def __init__(self, cfg: ProcessingConfig):
        self.cfg = cfg
        self.ks_unstable: Set[str] = set()
        self.pi_unstable: Set[str] = set()
        self.adversarial_unstable: Set[str] = set()
        self.all_features: Set[str] = set()
        
    def _load_json_results(self, path: Path) -> dict:
        """JSONファイルから結果を読み込む"""
        if not path.exists():
            raise FileNotFoundError(f"結果ファイルが見つかりません: {path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def load_all_results(self) -> None:
        """全ての検証結果を読み込む"""
        logger.info("=" * 60)
        logger.info("検証結果の読み込みを開始")
        logger.info("=" * 60)
        
        # KS検定結果
        logger.info("KS検定結果を読み込み中...")
        ks_data = self._load_json_results(self.cfg.ks_results_path)
        self.ks_unstable = set(ks_data['unstable_features'])
        logger.info(f"KS検定: {len(self.ks_unstable)}個の不安定特徴量を検出")
        
        # Permutation Importance結果
        logger.info("Permutation Importance結果を読み込み中...")
        pi_data = self._load_json_results(self.cfg.pi_results_path)
        self.pi_unstable = set(pi_data['unstable_features'])
        logger.info(f"Permutation Importance: {len(self.pi_unstable)}個の不安定特徴量を検出")
        
        # 敵対的検証結果
        logger.info("敵対的検証結果を読み込み中...")
        adv_data = self._load_json_results(self.cfg.adversarial_results_path)
        self.adversarial_unstable = set(adv_data['unstable_features'])
        logger.info(f"敵対的検証: {len(self.adversarial_unstable)}個の不安定特徴量を検出")
        
        # 全特徴量数を推定（いずれかのテストで検証された特徴量の総数）
        # 各テストの total_features_tested を確認
        total_features_ks = ks_data.get('total_features_tested', 0)
        total_features_pi = pi_data.get('total_features_tested', 0)
        total_features_adv = adv_data.get('total_features_tested', 0)
        
        # 最大値を採用（全テストで同じはずだが念のため）
        total_features = max(total_features_ks, total_features_pi, total_features_adv)
        
        logger.info(f"検証対象特徴量総数: {total_features}")
        logger.info("=" * 60)
    
    def merge_results(self) -> List[str]:
        """
        3つの検証結果を統合
        論理和（union）で不安定特徴量を統合
        """
        logger.info("検証結果を統合中...")
        
        # 3つのテストで検出された不安定特徴量の論理和
        total_unstable = self.ks_unstable.union(
            self.pi_unstable
        ).union(self.adversarial_unstable)
        
        logger.info(f"統合後の不安定特徴量数: {len(total_unstable)}")
        
        # 各検証での除外数をカウント（重複を含む）
        only_ks = self.ks_unstable - self.pi_unstable - self.adversarial_unstable
        only_pi = self.pi_unstable - self.ks_unstable - self.adversarial_unstable
        only_adv = self.adversarial_unstable - self.ks_unstable - self.pi_unstable
        
        ks_and_pi = (self.ks_unstable & self.pi_unstable) - self.adversarial_unstable
        ks_and_adv = (self.ks_unstable & self.adversarial_unstable) - self.pi_unstable
        pi_and_adv = (self.pi_unstable & self.adversarial_unstable) - self.ks_unstable
        
        all_three = self.ks_unstable & self.pi_unstable & self.adversarial_unstable
        
        logger.info("=" * 60)
        logger.info("詳細統計:")
        logger.info(f"  KS検定のみ: {len(only_ks)}")
        logger.info(f"  PI検定のみ: {len(only_pi)}")
        logger.info(f"  敵対的検証のみ: {len(only_adv)}")
        logger.info(f"  KS & PI: {len(ks_and_pi)}")
        logger.info(f"  KS & 敵対的: {len(ks_and_adv)}")
        logger.info(f"  PI & 敵対的: {len(pi_and_adv)}")
        logger.info(f"  全3つ: {len(all_three)}")
        logger.info("=" * 60)
        
        # 全特徴量数を推定（JSONファイルから取得）
        ks_data = self._load_json_results(self.cfg.ks_results_path)
        total_features = ks_data.get('total_features_tested', 0)
        
        # 安定特徴量のリストを生成（全特徴量の推定が必要）
        # ここでは不安定特徴量のみを保存し、安定特徴量数を計算
        stable_count = total_features - len(total_unstable)
        
        logger.info(f"安定特徴量数: {stable_count}")
        
        # 実際の安定特徴量リストを返すには全特徴量リストが必要
        # 現在の設計では不安定特徴量のみを記録
        # 下流工程で全特徴量から不安定特徴量を除外して使用
        
        return sorted(list(total_unstable))
    
    def save_results(self, unstable_features: List[str]) -> None:
        """統合結果を複数形式で保存"""
        logger.info("統合結果を保存中...")
        
        self.cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        # KS検定結果から全特徴量数を取得
        ks_data = self._load_json_results(self.cfg.ks_results_path)
        total_features = ks_data.get('total_features_tested', 0)
        stable_count = total_features - len(unstable_features)
        
        # JSON形式
        result_json = {
            "total_features": total_features,
            "stable_features_count": stable_count,
            "unstable_features": unstable_features,
            "validation_type": "first_defense_line_v3_merged",
            "excluded_by_ks": len(self.ks_unstable),
            "excluded_by_pi": len(self.pi_unstable),
            "excluded_by_adversarial": len(self.adversarial_unstable),
            "total_excluded": len(unstable_features),
            "timestamp": datetime.now().isoformat()
        }
        
        with open(self.cfg.stable_list_json, 'w', encoding='utf-8') as f:
            json.dump(result_json, f, indent=2, ensure_ascii=False)
        
        logger.info(f"JSON形式で保存: {self.cfg.stable_list_json}")
        
        # CSV形式（不安定特徴量リスト）
        unstable_df = pd.DataFrame({
            'feature_name': unstable_features,
            'status': 'unstable'
        })
        unstable_csv_path = self.cfg.artifacts_dir / "unstable_feature_list.csv"
        unstable_df.to_csv(unstable_csv_path, index=False)
        logger.info(f"CSV形式で保存: {unstable_csv_path}")
        
        # JOBLIB形式（不安定特徴量リスト）
        # 注: 下流工程では全特徴量から不安定特徴量を除外して安定特徴量を取得
        unstable_joblib_path = self.cfg.artifacts_dir / "unstable_feature_list.joblib"
        joblib.dump(unstable_features, unstable_joblib_path)
        logger.info(f"JOBLIB形式で保存: {unstable_joblib_path}")
        
        # adversarial scoresをコピー（参照用）
        if self.cfg.adversarial_scores_path.exists():
            import shutil
            shutil.copy(
                self.cfg.adversarial_scores_path,
                self.cfg.adversarial_scores_final
            )
            logger.info(f"敵対的重要度スコアをコピー: {self.cfg.adversarial_scores_final}")
        
        logger.info("=" * 60)
        logger.info("統合結果保存完了")
        logger.info(f"検証対象特徴量総数: {total_features}")
        logger.info(f"不安定特徴量数: {len(unstable_features)}")
        logger.info(f"安定特徴量数: {stable_count}")
        logger.info("=" * 60)


def main():
    """メイン実行"""
    cfg = ProcessingConfig()
    merger = ValidationResultMerger(cfg)
    
    try:
        merger.load_all_results()
        unstable_features = merger.merge_results()
        merger.save_results(unstable_features)
        
        logger.info("✅ 検証結果の統合が正常に完了しました")
        logger.info("\n次のステップ:")
        logger.info("  1. unstable_feature_list.joblib を使用して不安定特徴量を除外")
        logger.info("  2. 第二防衛線（SHAP評価）へ進む")
        
    except Exception as e:
        logger.error(f"❌ エラーが発生しました: {e}")
        raise


if __name__ == '__main__':
    main()