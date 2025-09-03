import logging
import sys
import os
from datetime import datetime

def setup_logger():
    """
    【v1.1 - プロセスID対応版】
    プロジェクト全体で利用するロガーを設定します。
    ログにプロセスIDを含めることで、並列処理の追跡を容易にします。
    """
    logger = logging.getLogger("ProjectForge")
    
    # ハンドラが既に追加されている場合は、再設定しない
    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.INFO)

    # --- ▼▼▼ ここを修正しました ▼▼▼ ---
    # フォーマッターにプロセスIDを追加
    formatter = logging.Formatter(
        '%(asctime)s - %(process)d - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    # --- ▲▲▲ ここまで修正 ▲▲▲ ---

    # コンソールハンドラ
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)

    # ファイルハンドラ
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"forge_run_{datetime.now().strftime('%Y%m%d')}.log")
    
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

# グローバルなloggerインスタンス
logger = setup_logger()