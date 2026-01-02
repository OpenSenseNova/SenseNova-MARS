#!/bin/bash
# Multi-node training for 8B model (N nodes, 8 GPUs per node)
#
# Usage:
#   On head node:    NODE_RANK=0 NNODES=<N> bash train_multi_node.sh
#   On worker nodes: NODE_RANK=<1..N-1> MASTER_ADDR=<head_ip> NNODES=<N> bash train_multi_node.sh
#
# Example (4 nodes):
#   Node 0 (head):   NODE_RANK=0 NNODES=4 bash train_multi_node.sh
#   Node 1 (worker): NODE_RANK=1 MASTER_ADDR=10.0.0.1 NNODES=4 bash train_multi_node.sh
#   Node 2 (worker): NODE_RANK=2 MASTER_ADDR=10.0.0.1 NNODES=4 bash train_multi_node.sh
#   Node 3 (worker): NODE_RANK=3 MASTER_ADDR=10.0.0.1 NNODES=4 bash train_multi_node.sh

set -ex

# ==================== USER CONFIGURATION (Edit these) ====================
export TEXT_SEARCH_ADDRESS="<INFRA_SERVER_IP>:8000"
export LOCAL_DATABASE_ADDRESS="<INFRA_SERVER_IP>:8001"
export AZURE_OPENAI_API_KEY="<YOUR_AZURE_OPENAI_KEY>"
export WANDB_API_KEY="<YOUR_WANDB_API_KEY>"
LLM_JUDGE_URL="<LLM_JUDGE_SERVER_IP>:8181"
# ==========================================================================

cd $(dirname $0)

num_nodes=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_PORT=${MASTER_PORT:-6379}

# Scale agent workers with number of nodes
AGENT_WORKERS=$((32 * num_nodes))

# Auto-detect MASTER_ADDR on head node
if [ "$NODE_RANK" -eq 0 ] && [ -z "$MASTER_ADDR" ]; then
    MASTER_ADDR=$(hostname -I | awk '{print $1}')
fi

if [ -z "$MASTER_ADDR" ]; then
    echo "ERROR: MASTER_ADDR must be set for worker nodes"
    exit 1
fi

# ==================== Environment Variables ====================
export CODE_DIR=$(dirname $0)/verl
export PYTHONPATH=$CODE_DIR:$PYTHONPATH

export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0

export MAX_PIXELS=8294400
export MIN_PIXELS=65536
export NCCL_IB_TIMEOUT=200
export NCCL_IB_RETRY_CNT=20

# Disable DeepGEMM to avoid TMA descriptor issues with vision encoder's variable matrix sizes
export SGL_ENABLE_JIT_DEEPGEMM=0

# ==================== Ray Setup ====================
ray stop --force 2>/dev/null || true
sleep 3

if [ "$NODE_RANK" -eq 0 ]; then
    # Head node
    OBJECT_STORE_MEM=$(free -b | awk '/^Mem:/{printf "%.0f", $2 * 0.4}')
    ray start --head --port=${MASTER_PORT} --dashboard-host=0.0.0.0 --dashboard-port=8265 \
        --object-store-memory=$OBJECT_STORE_MEM --disable-usage-stats
    sleep 30

    # Wait for all nodes
    for ((i=1; i<=30; i++)); do
        CONNECTED=$(ray status 2>/dev/null | grep -c "node_" || echo "0")
        echo "Connected nodes: $CONNECTED / $num_nodes"
        [ "$CONNECTED" -ge "$num_nodes" ] && break
        sleep 30
    done
    ray status

    # ==================== Training ====================
    MODEL_PATH="Qwen/Qwen3-VL-8B-Instruct"

    TOOL_CONFIG_TRAIN="$(dirname $0)/config/tool_config/tools_train.yaml"
    TOOL_CONFIG_VAL="$(dirname $0)/config/tool_config/tools_val.yaml"

    train_files=$(dirname $0)/train_qwen3_vl_8b.json
    val_files=$(dirname $0)/test_subset.json

    model_name=$(basename "$MODEL_PATH")
    # Fixed experiment name for checkpoint matching (no timestamp)
    exp_name=sensesearch_${model_name}_lr1e-5
    # Timestamp for log file only
    LOG_TIMESTAMP=$(date +%Y%m%d_%H%M%S)

    mkdir -p $(dirname $0)/rollout_data/$exp_name/train
    mkdir -p $(dirname $0)/rollout_data/$exp_name/validation
    mkdir -p $(dirname $0)/logs

    python3 -m verl.trainer.main_ppo \
        algorithm.adv_estimator=grpo \
        algorithm.use_kl_in_reward=False \
        data.custom_cls.path="pkg://verl.utils.dataset.rl_dataset_json_v2" \
        data.custom_cls.name="RLHFJSONDatasetV2" \
        data.train_files=$train_files \
        data.val_files=$val_files \
        data.train_batch_size=128 \
        data.val_batch_size=512 \
        data.dataloader_num_workers=8 \
        data.max_prompt_length=$((1024 * 16)) \
        data.max_response_length=$((1024 * 16)) \
        data.filter_overlong_prompts=False \
        data.truncation='error' \
        data.return_raw_chat=True \
        data.return_multi_modal_inputs=False \
        data.image_key=image \
        data.image_patch_size=16 \
        data.tool_config_path="${TOOL_CONFIG_TRAIN}" \
        +data.val_tool_config_path="${TOOL_CONFIG_VAL}" \
        actor_rollout_ref.model.path=${MODEL_PATH} \
        actor_rollout_ref.actor.optim.lr=1e-5 \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.use_dynamic_bsz=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=128 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((1024 * 16)) \
        actor_rollout_ref.actor.use_kl_loss=False \
        actor_rollout_ref.actor.kl_loss_coef=0.0 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.actor.entropy_coeff=0 \
        actor_rollout_ref.actor.clip_ratio_low=0.2 \
        actor_rollout_ref.actor.clip_ratio_high=0.28 \
        actor_rollout_ref.actor.ulysses_sequence_parallel_size=2 \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.model.trust_remote_code=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
        actor_rollout_ref.actor.freeze_vision_tower=True \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
        actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=32768 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
        actor_rollout_ref.rollout.name=sglang \
        actor_rollout_ref.rollout.mode=async \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
        actor_rollout_ref.rollout.enable_chunked_prefill=True \
        actor_rollout_ref.rollout.enforce_eager=False \
        actor_rollout_ref.rollout.max_num_batched_tokens=$((1024 * 32)) \
        actor_rollout_ref.rollout.max_model_len=$((1024 * 32)) \
        actor_rollout_ref.rollout.max_num_seqs=32 \
        actor_rollout_ref.rollout.response_length=$((1024 * 16)) \
        actor_rollout_ref.rollout.n=8 \
        +actor_rollout_ref.rollout.seed=3407 \
        +actor_rollout_ref.rollout.repetition_penalty=1.0 \
        +actor_rollout_ref.rollout.presence_penalty=1.5 \
        actor_rollout_ref.rollout.calculate_log_probs=True \
        algorithm.rollout_correction.bypass_mode=True \
        actor_rollout_ref.rollout.free_cache_engine=True \
        +actor_rollout_ref.rollout.engine_kwargs.sglang.enable_deterministic_inference=True \
        actor_rollout_ref.rollout.val_kwargs.n=1 \
        actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
        actor_rollout_ref.rollout.val_kwargs.top_p=0.8 \
        actor_rollout_ref.rollout.val_kwargs.top_k=20 \
        actor_rollout_ref.rollout.val_kwargs.do_sample=True \
        +actor_rollout_ref.rollout.val_kwargs.repetition_penalty=1.0 \
        +actor_rollout_ref.rollout.val_kwargs.presence_penalty=1.5 \
        +actor_rollout_ref.rollout.val_kwargs.seed=3407 \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
        actor_rollout_ref.ref.fsdp_config.param_offload=False \
        actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
        actor_rollout_ref.rollout.multi_turn.enable=True \
        actor_rollout_ref.rollout.multi_turn.max_assistant_turns=10 \
        actor_rollout_ref.rollout.multi_turn.max_tool_response_length=8192 \
        actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side="right" \
        actor_rollout_ref.rollout.multi_turn.tool_config_path="${TOOL_CONFIG_TRAIN}" \
        +actor_rollout_ref.rollout.multi_turn.val_tool_config_path="${TOOL_CONFIG_VAL}" \
        actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
        actor_rollout_ref.rollout.agent.num_workers=$AGENT_WORKERS \
        reward_model.reward_manager=tool \
        reward_model.launch_reward_fn_async=True \
        +reward_model.reward_kwargs.log_num_round=True \
        +reward_model.reward_kwargs.format_score=0.5 \
        +reward_model.reward_kwargs.llm_judge_model='Qwen3-VL-32B-Instruct' \
        +reward_model.reward_kwargs.llm_judge_urls=["${LLM_JUDGE_URL}"] \
        +reward_model.reward_kwargs.llm_judge_timeout=300 \
        +reward_model.reward_kwargs.llm_judge_concurrency_limit=4 \
        +reward_model.val_reward_kwargs.log_num_round=True \
        +reward_model.val_reward_kwargs.format_score=0.5 \
        +reward_model.val_reward_kwargs.llm_judge_model='gpt-4o-2024-11-20' \
        +reward_model.val_reward_kwargs.llm_judge_timeout=300 \
        +reward_model.val_reward_kwargs.llm_judge_concurrency_limit=2 \
        trainer.critic_warmup=0 \
        trainer.val_before_train=False \
        trainer.resume_mode="auto" \
        trainer.logger=['console','wandb'] \
        trainer.project_name='sensesearch_rl' \
        trainer.experiment_name=$exp_name \
        trainer.n_gpus_per_node=8 \
        trainer.nnodes=$num_nodes \
        trainer.save_freq=10 \
        trainer.test_freq=10 \
        trainer.rollout_data_dir=$(dirname $0)/rollout_data/$exp_name/train \
        trainer.validation_data_dir=$(dirname $0)/rollout_data/$exp_name/validation \
        trainer.total_epochs=20 2>&1 | tee $(dirname $0)/logs/${exp_name}_${LOG_TIMESTAMP}.log

else
    # Worker node
    sleep 30
    OBJECT_STORE_MEM=$(free -b | awk '/^Mem:/{printf "%.0f", $2 * 0.4}')
    for ((i=1; i<=10; i++)); do
        ray start --address="${MASTER_ADDR}:${MASTER_PORT}" \
            --object-store-memory=$OBJECT_STORE_MEM --disable-usage-stats && break
        sleep 30
    done
    sleep infinity
fi
