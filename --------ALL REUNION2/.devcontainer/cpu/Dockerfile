# Dockerfile - CPU専用環境
FROM python:3.11-slim

# タイムゾーンの設定
ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# 新しいユーザー'hephaestus-zero'を作成し、sudoグループに追加
ARG USERNAME=hephaestus-zero
RUN useradd -m -s /bin/bash -G sudo ${USERNAME} && \
    # sudoers.dディレクトリを作成
    mkdir -p /etc/sudoers.d && \
    # パスワードなしでsudoコマンドを使えるように設定
    echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME}

# 必要なシステムパッケージのインストール
# (scipy, scikit-learn, tensorflowなどのビルドに必要)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    pkg-config \
    git \
    curl \
    tree \
    sudo \
    procps \
    tmux \
    htop \
    nano \
    wget \
    lsof \
    && rm -rf /var/lib/apt/lists/*

# 作業ディレクトリの設定
WORKDIR /workspace

# requirements-cpu.txtをコピーしてインストール
COPY requirements-cpu.txt .
RUN pip install --no-cache-dir -r requirements-cpu.txt

# /home/${USERNAME}/.jupyter に設定ファイルを作成するように変更
USER ${USERNAME}
RUN mkdir -p /home/${USERNAME}/.jupyter && \
    echo "c.ServerApp.token = ''" > /home/${USERNAME}/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.password = ''" >> /home/${USERNAME}/.jupyter/jupyter_lab_config.py
USER root

# ポートの公開
EXPOSE 8888 8787

# デフォルトコマンド
CMD ["bash"]