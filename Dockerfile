FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Set non-interactive timezone
ENV DEBIAN_FRONTEND=noninteractive

# Install basic dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Rust Toolchain
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Setup application directory
WORKDIR /app

# Create a virtual environment and ensure it activates for all subsequent RUN/CMD
RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and bootstrap Python toolchain
COPY requirements.txt .
RUN pip install --no-cache-dir -U pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

# Copy complete project code
COPY . .

# Environment variables needed for seamless torch extension linking inside maturin/setuptools
ENV LIBTORCH_USE_PYTORCH=1
ENV LIBTORCH_BYPASS_VERSION_CHECK=1
ENV _GLIBCXX_USE_CXX11_ABI=1

# Compile and install the Rust and CUDA backend into standard site-packages
RUN pip install --no-cache-dir .

EXPOSE 8000

ENTRYPOINT ["tricked"]
