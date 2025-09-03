# ===================================================================
# Stage 1: Builder
# このステージでは、すべての依存関係をインストールし、
# クリーンなconda環境を構築します。
# ===================================================================
FROM rapidsai/base:25.08-cuda12.0-py3.11

# システム依存関係のインストール（必要に応じて）
# 例: C++コンパイラなど
USER root
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
USER rapids

# 環境定義ファイルをコンテナにコピー
WORKDIR /tmp
COPY environment.yml.

# mambaを使用してconda環境を構築
# これにより、すべてのライブラリがキャッシュと共にインストールされる
RUN mamba env create -f environment.yml

# ===================================================================
# Stage 2: Final Image
# このステージでは、builderで構築したクリーンなconda環境のみをコピーし、
# 軽量でセキュアな最終イメージを作成します。
# ===================================================================
FROM rapidsai/base:25.08-cuda12.0-py3.11

# builderステージから完成したconda環境をコピー
COPY --from=builder /opt/conda/envs/ss-class-env /opt/conda/envs/ss-class-env

# 環境変数を設定して、新しい環境をデフォルトにする
ENV CONDA_DEFAULT_ENV=ss-class-env
ENV CONDA_PREFIX=/opt/conda/envs/ss-class-env
ENV PATH=$CONDA_PREFIX/bin:$PATH

# NVIDIAドライバケーパビリティの設定
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# 作業ディレクトリの設定
WORKDIR /workspace

# デフォルトのコマンド
CMD ["/bin/bash"]