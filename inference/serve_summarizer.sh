
CUDA_VISIBLE_DEVICES=2,3 python3 -m sglang.launch_server \
  --model-path Qwen/Qwen3-32B \
  --served-model-name "qwen3-32b" \
  --tp-size 2 \
  --dtype bfloat16 \
  --host 0.0.0.0 --port 8181 \
  --mem-fraction-static 0.6
