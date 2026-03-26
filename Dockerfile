# Base Image: Official PyTorch Devel Image with CUDA 12.4
# (Pre-packaged with PyTorch, CuDNN, and all C++ headers! Bypasses all pip timeouts and cxx11 ABI linkage errors)
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive

# Install System Dependencies and Node.js
RUN apt-get update && apt-get install -y curl build-essential wget && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*

# Install Rust toolchain natively
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /app

# Copy dependency files first for caching
COPY pyproject.toml .
COPY scripts/ /app/scripts/
COPY conf/ /app/conf/
COPY src/ /app/src/

# Install maturin to build Rust bindings natively (PyTorch is already installed via base image!)
RUN pip install --no-cache-dir setuptools wheel maturin
RUN pip install --no-cache-dir fastapi "uvicorn[standard]" websockets pydantic redis

# Compile the Rust Extension (`tricked_rs`)
WORKDIR /app/src/tricked_rs
ENV LIBTORCH_USE_PYTORCH=1
ENV LIBTORCH_BYPASS_VERSION_CHECK=1
ENV PYTHONPATH=/app/src
RUN rm -rf target/wheels/* && maturin build --release --manifest-path Cargo.toml

# Dynamically wrap LD_LIBRARY_PATH to compile the Rust binary cleanly
RUN export LD_LIBRARY_PATH="$(python -c 'import torch; import os; print(os.path.dirname(torch.__file__) + "/lib")'):$LD_LIBRARY_PATH" && \
    cargo build --release --bin self_play_worker

RUN pip install target/wheels/*.whl

WORKDIR /app
# Install the root `tricked` python project
RUN pip install -e .

WORKDIR /app
# Copy orchestration scripts
COPY docker-entrypoint.sh /app/
RUN chmod +x /app/docker-entrypoint.sh


EXPOSE 8080 6006 5173

ENTRYPOINT ["/app/docker-entrypoint.sh"]
