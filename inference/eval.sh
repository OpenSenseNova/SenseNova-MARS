#!/bin/bash

set -e


# Model server
export MODEL_BASE_URL="http://localhost:8888"

# Judge - fill in one of the following:
# For --judge-client azure:
export AZURE_OPENAI_API_KEY="<YOUR_AZURE_OPENAI_API_KEY>"
export AZURE_OPENAI_BASE_URL="<YOUR_AZURE_OPENAI_BASE_URL>"
export AZURE_API_VERSION="2025-01-01-preview"
# For --judge-client openai:
export OPENAI_API_KEY="<YOUR_OPENAI_API_KEY>"

# Web search
export SERPER_API_KEY="<YOUR_SERPER_API_KEY>"

# Summarizer
export SUMMARIZER_BASE_URL="http://localhost:8181"
export SUMMARIZER_MODEL="qwen3-32b"

python eval.py \
    --model-client openai \
    --judge-client azure \
    --model SenseNova-MARS-32B \
    --mode tool \
    --datasets ../test_subset.json \
    --data-root .. \
    --tool-config tools_eval.yaml \
    --max-concurrent 64 \
    --max-turns 50 \
    --serper-concurrency 64 \
    # --search-cache-dir .
