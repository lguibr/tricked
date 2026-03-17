# Base Image: Unbuntu with CUDA 12.4 + cuDNN (Required for GPU accelerated PyTorch/MuZero)
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# Prevent interactive prompts during apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Install System Dependencies & Python 3.10+
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install Rust toolchain via rustup (Required for PyO3 Rust extension compilation)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /app

# Copy dependency files first for caching
COPY pyproject.toml .
# Do not copy lock files if we don't have pdm/poetry configured, but let's copy the entire src/ 
COPY src/ /app/src/

# Install base python dependencies required for Rust compilation
RUN pip3 install --no-cache-dir setuptools wheel maturin
RUN pip3 install --no-cache-dir torch tensorboard numpy flask

# Compile the Rust Extension (`tricked_rs`) natively inside Docker
WORKDIR /app/src/tricked_rs
RUN maturin build --release --manifest-path Cargo.toml
RUN pip3 install target/wheels/tricked_rs-*.whl

WORKDIR /app
# Install the root `tricked` python project
RUN pip3 install -e .

# Copy orchestration scripts
COPY docker-entrypoint.sh /app/
RUN chmod +x /app/docker-entrypoint.sh

# Create runs/ directory so TensorBoard binds successfully even if empty
RUN mkdir -p /app/runs/tricked_muzero

# Expose Web UI (8080) and TensorBoard (6006)
EXPOSE 8080 6006

# Command starts the Triple-Daemon bash script
ENTRYPOINT ["/app/docker-entrypoint.sh"]
