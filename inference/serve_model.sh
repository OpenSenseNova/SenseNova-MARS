
export SGLANG_VLM_CACHE_SIZE_MB=4096
CUDA_VISIBLE_DEVICES=0,1 python -m sglang.launch_server \
  --model-path sensenova/SenseNova-MARS-32B \
  --host 0.0.0.0 --port 8888 \
  --dtype bfloat16 \
  --served-model-name SenseNova-MARS-32B \
  --tp 2 \
  --mem-fraction-static 0.6 \
  --enable-deterministic-inference
