#!/bin/bash
# Single-node evaluation (1 node, 8 GPUs)

set -ex

# ==================== USER CONFIGURATION (Edit these) ====================
export MODEL_PATH="<YOUR_TRAINED_MODEL_PATH>"
export TEXT_SEARCH_ADDRESS="<INFRA_SERVER_IP>:8000"
export LOCAL_DATABASE_ADDRESS="<INFRA_SERVER_IP>:8001"
export AZURE_OPENAI_API_KEY="<YOUR_AZURE_OPENAI_KEY>"
# ==========================================================================

num_nodes=1

# ==================== Environment Variables ====================
export CODE_DIR=$(dirname $0)/verl
export PYTHONPATH=$CODE_DIR:$PYTHONPATH

export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export MAX_PIXELS=8294400
export MIN_PIXELS=65536

# ==================== Ray Setup ====================
ray stop --force 2>/dev/null || true
sleep 3
pkill -9 -f "ray::" 2>/dev/null || true
pkill -9 -f "raylet" 2>/dev/null || true
sleep 2

ray start --head --num-gpus=8 --dashboard-host=0.0.0.0 --dashboard-port=8265 --disable-usage-stats
sleep 5
ray status || { echo "ERROR: Ray failed to start"; exit 1; }

# ==================== Evaluation ====================
TOOL_CONFIG_TRAIN="$(dirname $0)/config/tool_config/tools_train.yaml"
TOOL_CONFIG_VAL="$(dirname $0)/config/tool_config/tools_train.yaml"

train_files=$(dirname $0)/train_qwen3_vl_8b.json
val_files=$(dirname $0)/test_all.json

model_name=$(basename "$MODEL_PATH")
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
exp_name=sensesearch_eval_${model_name}_${TIMESTAMP}

mkdir -p $(dirname $0)/rollout_data/$exp_name/train
mkdir -p $(dirname $0)/rollout_data/$exp_name/validation
mkdir -p $(dirname $0)/logs

LOG_TIMESTAMP=$(date +%Y%m%d_%H%M%S)

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
    actor_rollout_ref.actor.optim.lr=5e-6 \
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
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=65536 \
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
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
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
    actor_rollout_ref.rollout.multi_turn.tool_config_path="${TOOL_CONFIG_TRAIN}" \
    +actor_rollout_ref.rollout.multi_turn.val_tool_config_path="${TOOL_CONFIG_VAL}" \
    actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
    actor_rollout_ref.rollout.agent.num_workers=16 \
    reward_model.reward_manager=tool \
    reward_model.launch_reward_fn_async=True \
    +reward_model.val_reward_kwargs.log_num_round=True \
    +reward_model.val_reward_kwargs.format_score=0.5 \
    +reward_model.val_reward_kwargs.llm_judge_model='gpt-4o-2024-11-20' \
    +reward_model.val_reward_kwargs.llm_judge_concurrency_limit=2 \
    +reward_model.val_reward_kwargs.llm_judge_timeout=300 \
    trainer.critic_warmup=0 \
    trainer.val_before_train=True \
    trainer.resume_mode='disable' \
    trainer.logger=['console'] \
    trainer.project_name='sensesearch_rl' \
    trainer.experiment_name=$exp_name \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=$num_nodes \
    trainer.save_freq=10 \
    trainer.test_freq=0 \
    trainer.rollout_data_dir=$(dirname $0)/rollout_data/$exp_name/train \
    trainer.validation_data_dir=$(dirname $0)/rollout_data/$exp_name/validation \
    trainer.total_epochs=0 2>&1 | tee $(dirname $0)/logs/${exp_name}.log
