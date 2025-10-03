# Dockerfile - CPU専用環境
FROM python:3.11-slim

# タイムゾーンの設定
ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# --- ロケール設定 (警告回避のため) ---
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

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
    && rm -rf /var/lib/apt/lists/*

# 作業ディレクトリの設定
WORKDIR /workspace

# requirements-cpu.txtをコピーしてインストール
COPY requirements-cpu.txt .
RUN pip install --no-cache-dir -r requirements-cpu.txt

# JupyterLabのデフォルト設定
RUN mkdir -p /root/.jupyter && \
    echo "c.ServerApp.token = ''" > /root/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.password = ''" >> /root/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.allow_root = True" >> /root/.jupyter/jupyter_lab_config.py

# ポートの公開
EXPOSE 8888 8787

# デフォルトコマンド
CMD ["bash"]