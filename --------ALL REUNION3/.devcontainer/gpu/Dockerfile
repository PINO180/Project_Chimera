# Dockerfile - GPU環境 (RAPIDS) - 最終修正版
FROM rapidsai/base:25.08-cuda12.0-py3.11

# --- ↓↓↓ここから修正↓↓↓ ---
# rootユーザーに切り替えて、権限が必要な操作を実行
USER root

# タイムゾーンの設定
ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# システムライブラリのインストール (sudoを追加)
RUN apt-get update && apt-get install -y --no-install-recommends \
    sudo \
    tree \
    && rm -rf /var/lib/apt/lists/*

# 新しいユーザー'hermes_talaria'を作成
ARG USERNAME=hermes_talaria
RUN useradd -m -s /bin/bash -G sudo ${USERNAME} && \
    mkdir -p /etc/sudoers.d && \
    echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME}
# --- ↑↑↑ここまで修正↑↑↑ ---


# --- プロジェクト設定 ---
WORKDIR /workspace

# --- Pythonライブラリのインストール ---
COPY requirements-gpu.txt .
RUN pip install --no-cache-dir -r requirements-gpu.txt

# --- JupyterLabの設定 ---
# hermes_talariaユーザーに切り替えて設定ファイルを作成
USER ${USERNAME}
RUN mkdir -p /home/${USERNAME}/.jupyter && \
    echo "c.ServerApp.token = ''" > /home/${USERNAME}/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.password = ''" >> /home/${USERNAME}/.jupyter/jupyter_lab_config.py
# 後続のコマンドのためにrootユーザーに戻す
USER root

# --- ポートの公開 ---
EXPOSE 8888 8787

# --- デフォルトコマンド ---
CMD ["bash"]