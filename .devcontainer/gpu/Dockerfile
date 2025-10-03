# Dockerfile - GPU環境 (RAPIDS)
FROM rapidsai/base:25.08-cuda12.0-py3.11

# タイムゾーンの設定
ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# 作業ディレクトリの設定
WORKDIR /workspace

# requirements-gpu.txtをコピーしてインストール
COPY requirements-gpu.txt .
RUN pip install --no-cache-dir -r requirements-gpu.txt

# JupyterLabのデフォルト設定
RUN mkdir -p /root/.jupyter && \
    echo "c.ServerApp.token = ''" > /root/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.password = ''" >> /root/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.allow_root = True" >> /root/.jupyter/jupyter_lab_config.py

# ポートの公開
EXPOSE 8888 8787

# デフォルトコマンド
CMD ["bash"]