# 1. Base Image
FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-devel

# =============================================================================
# [OPTIONAL: REGIONAL NETWORK OPTIMIZATION]
# To enable, use: --build-arg USE_MIRROR=true during build.
# =============================================================================
ARG USE_MIRROR=false

RUN if [ "$USE_MIRROR" = "true" ]; then \
    echo "Applying regional mirrors..." && \
    sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list && \
    sed -i 's/security.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list && \
    pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/ && \
    pip config set global.trusted-host mirrors.aliyun.com; \
    fi

ENV PLAYWRIGHT_DOWNLOAD_HOST=${USE_MIRROR:+"https://npmmirror.com/mirrors/playwright/"}
ENV HF_ENDPOINT=${USE_MIRROR:+"https://hf-mirror.com"}
# =============================================================================

# 2. Set Build Variables
ENV TORCH_CUDA_ARCH_LIST="9.0"
ENV MAX_JOBS=4 
ENV FLASH_ATTENTION_FORCE_BUILD=TRUE
ENV HOME=/root
ENV LD_LIBRARY_PATH="/usr/local/cuda-12.9/compat:$LD_LIBRARY_PATH"

# 3. System dependencies
RUN apt-get update && apt-get install -y \
    vim tmux wget curl git build-essential openssh-server \
    ninja-build cuda-compat-12-9 \
    && rm -rf /var/lib/apt/lists/*

# 4. Basic Python Packages
RUN pip3 install --no-cache-dir \
    pandas tensordict omegaconf hydra-core --upgrade \
    torchdata codetiming datasets func_timeout pylatexenc \
    sandbox_fusion peft wandb tensorboard liger-kernel \
    tencentcloud-sdk-python "ray[default]" lxml beautifulsoup4 \
    aiohttp shortuuid ninja

# 5. Playwright
RUN pip3 install --no-cache-dir playwright playwright-stealth && \
    playwright install-deps && \
    playwright install

# 6. Core AI Packages
RUN pip3 install --no-cache-dir \
    qwen_vl_utils vllm==0.12.0 transformers==4.57.3 flashinfer-python

# 7. Flash Attention (Build from Source - v2.8.3)
# We use --branch and --depth 1 for a faster, specific checkout
RUN git clone https://github.com/Dao-AILab/flash-attention.git \
    --branch v2.8.3 --single-branch --depth 1 /tmp/flash-attention && \
    cd /tmp/flash-attention && \
    python3 setup.py install && \
    rm -rf /tmp/flash-attention

RUN pip install sglang==0.5.6
RUN pip install nvidia-cudnn-cu12==9.16.0.29

# 8. Final Cleanup
RUN pip cache purge
WORKDIR /workspace
CMD ["/bin/bash"]
