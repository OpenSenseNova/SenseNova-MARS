# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Tool Reward Manager for VLM multi-turn tool training.

Supports dataset-driven reward configuration:
- reward_fn: List of scoring functions used for training reward
- unused_reward_fn: List of scoring functions used for metrics only (not reward)

Scoring functions available:
- f1_score: F1 score for QA
- em_score: Exact match score
- em_score_mcq: Exact match for multiple choice
- llm_score: LLM judge score
- format_score: Format validation score
"""

import logging
import os
import random
from collections import defaultdict

import openai
import torch
from openai import AzureOpenAI

from verl import DataProto
from verl.utils.reward_score.tool import (
    compute_score_fns,
    get_judge_llm_stats,
    configure_judge_llm_semaphore,
)
from verl.workers.reward_manager.registry import register

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("tool")
class ToolRewardManager:
    """Reward manager for VLM multi-turn tool training.

    Reads reward_fn and unused_reward_fn from dataset configuration to compute
    training rewards and evaluation metrics.
    """

    _init_logged_train = False  # Class-level flag to log init info only once for train
    _init_logged_val = False  # Class-level flag to log init info only once for val

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key="data_source",
        max_turns=10,
        **kwargs
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score_fns
        self.reward_fn_key = reward_fn_key
        self.max_turns = max_turns

        self.is_val = kwargs.get("is_val", False)
        self.log_num_round = kwargs.get("log_num_round", False)

        # Format score weights
        self.format_score = kwargs.get("format_score", 0.5)

        # LLM judge configuration
        llm_judge_model = kwargs.get("llm_judge_model")
        llm_judge_urls = kwargs.get("llm_judge_urls", [])
        llm_judge_temperature = kwargs.get("llm_judge_temperature", 0.0)
        llm_judge_max_tokens = kwargs.get("llm_judge_max_tokens", 8192)
        llm_judge_timeout = kwargs.get("llm_judge_timeout", 120)
        llm_judge_is_vision_model = kwargs.get("llm_judge_is_vision_model", True)
        llm_judge_enable_thinking = kwargs.get("llm_judge_enable_thinking", False)
        llm_judge_concurrency_limit = kwargs.get("llm_judge_concurrency_limit", 0)  # 0 = no limit

        # Configure semaphore for rate limiting LLM judge API calls
        # This limits concurrent requests per worker process
        # Total max concurrent = num_reward_workers * llm_judge_concurrency_limit
        if llm_judge_concurrency_limit > 0:
            configure_judge_llm_semaphore(llm_judge_concurrency_limit)

        # Auto-detect provider and create appropriate clients
        if llm_judge_model:
            is_azure = "gpt-4o" in llm_judge_model.lower()

            if is_azure:
                # GPT models -> Azure OpenAI
                llm_judge_api_key = kwargs.get("llm_judge_api_key", os.getenv("AZURE_OPENAI_API_KEY"))
                azure_endpoint = kwargs.get("llm_judge_azure_endpoint", "https://duomotai.openai.azure.com/")
                api_version = kwargs.get("llm_judge_azure_api_version", "2025-01-01-preview")

                self.clients = [
                    AzureOpenAI(
                        api_key=llm_judge_api_key,
                        azure_endpoint=azure_endpoint,
                        api_version=api_version,
                        max_retries=5,
                    )
                ]
                self.llm_judge_urls = []
                # Log judge config for train/val separately
                should_log = (self.is_val and not ToolRewardManager._init_logged_val) or \
                             (not self.is_val and not ToolRewardManager._init_logged_train)
                if should_log:
                    mode = "VAL" if self.is_val else "TRAIN"
                    print(f"[ToolRewardManager] [{mode}] llm_judge_model={llm_judge_model}")
                    print(f"[ToolRewardManager] [{mode}] Using Azure OpenAI for {llm_judge_model}")
                    if self.is_val:
                        ToolRewardManager._init_logged_val = True
                    else:
                        ToolRewardManager._init_logged_train = True

            else:
                # Non-GPT models -> vLLM (requires URLs)
                if not llm_judge_urls:
                    raise ValueError(
                        f"Model '{llm_judge_model}' requires llm_judge_urls.\n"
                        f"Example: +reward_model.reward_kwargs.llm_judge_urls=['10.119.27.50:8181']"
                    )

                llm_judge_api_key = kwargs.get("llm_judge_api_key", "token-abc123")
                self.clients = [
                    openai.OpenAI(api_key=llm_judge_api_key, base_url=f"http://{url}/v1", max_retries=5)
                    for url in llm_judge_urls
                ]
                self.llm_judge_urls = llm_judge_urls
                # Log judge config for train/val separately
                should_log = (self.is_val and not ToolRewardManager._init_logged_val) or \
                             (not self.is_val and not ToolRewardManager._init_logged_train)
                if should_log:
                    mode = "VAL" if self.is_val else "TRAIN"
                    print(f"[ToolRewardManager] [{mode}] llm_judge_model={llm_judge_model}")
                    print(f"[ToolRewardManager] [{mode}] Using vLLM at {llm_judge_urls}")
                    if self.is_val:
                        ToolRewardManager._init_logged_val = True
                    else:
                        ToolRewardManager._init_logged_train = True
        else:
            self.clients = []
            self.llm_judge_urls = []

        # Package config for scoring functions
        if llm_judge_model:
            self.llm_judge_config = {
                "model": llm_judge_model,
                "temperature": llm_judge_temperature,
                "max_tokens": llm_judge_max_tokens,
                "timeout": llm_judge_timeout,
                "is_vision_model": llm_judge_is_vision_model,
                "enable_thinking": llm_judge_enable_thinking,
            }
        else:
            self.llm_judge_config = None

        # LLM judge success rate threshold (0.0-1.0, crash if below)
        # Set to 0 to disable the check
        self.judge_llm_success_threshold = kwargs.get("judge_llm_success_threshold", 0.95)

    def __call__(self, data: DataProto, return_dict: bool = False):
        """Compute rewards for the batch.

        Args:
            data: DataProto containing batch data
            return_dict: If True, return dict with reward_tensor and reward_extra_info

        Returns:
            reward_tensor or dict with reward_tensor and reward_extra_info
        """
        # If rm_scores already exist, use pre-computed metrics from parallel generation
        if "rm_scores" in data.batch.keys():
            reward_tensor = data.batch["rm_scores"]

            # Collect all reward_fn and unused_reward_fn names
            all_metric_names = set()
            for i in range(len(data)):
                data_item = data[i]
                reward_fn_names = data_item.non_tensor_batch.get("reward_model", {}).get("reward_fn", [])
                unused_reward_fn_names = data_item.non_tensor_batch.get("reward_model", {}).get("unused_reward_fn", [])
                all_metric_names.update(reward_fn_names)
                all_metric_names.update(unused_reward_fn_names)

            # Check if metrics were pre-computed during parallel generation
            # agent_loop.py stores them directly in non_tensor_batch with metric names as keys
            reward_extra_info = {}
            for name in all_metric_names:
                if name in data.non_tensor_batch:
                    values = data.non_tensor_batch[name]
                    reward_extra_info[name] = values.tolist() if hasattr(values, 'tolist') else list(values)

            if reward_extra_info:
                if return_dict:
                    logger.debug(f"rm_scores exists, using pre-computed metrics: {list(reward_extra_info.keys())}")
                    return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
                else:
                    return reward_tensor

            # Fallback: no pre-computed metrics found, return empty
            if return_dict:
                logger.debug("rm_scores exists, no pre-computed metrics found")
                return {"reward_tensor": reward_tensor, "reward_extra_info": {}}
            else:
                return reward_tensor

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        def _process_item(i, data_item):
            """Process a single data item."""
            try:
                prompt_ids = data_item.batch["prompts"]
                prompt_length = prompt_ids.shape[-1]

                valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()

                response_ids = data_item.batch["responses"]
                valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
                valid_response_ids = response_ids[:valid_response_length]

                # Decode response
                valid_response_str = self.tokenizer.decode(valid_response_ids)

                ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
                reward_fn_names = data_item.non_tensor_batch["reward_model"]["reward_fn"]
                unused_reward_fn_names = data_item.non_tensor_batch["reward_model"]["unused_reward_fn"]


                raw_prompt = data_item.non_tensor_batch.get("raw_prompt")
                multi_modal_inputs = data_item.non_tensor_batch.get("multi_modal_inputs", None)
                multi_modal_data = data_item.non_tensor_batch.get("multi_modal_data", None)

                score = {}

                # Common kwargs for scoring functions
                common_kwargs = dict(
                    clients=self.clients,
                    llm_config=self.llm_judge_config,
                    raw_prompt=raw_prompt,
                    multi_modal_inputs=multi_modal_inputs,
                    multi_modal_data=multi_modal_data,
                    max_turns=self.max_turns,
                    eos_token=self.tokenizer.eos_token,
                    data_non_tensor_batch=data_item.non_tensor_batch,
                    is_val=self.is_val,
                    format_score=self.format_score,
                )

                # Pass is_valid_format if provided by agent_loop
                if 'is_valid_format' in data_item.non_tensor_batch:
                    common_kwargs['is_valid_format'] = data_item.non_tensor_batch['is_valid_format']

                # Compute reward scores
                total_score = 0.0
                for name in reward_fn_names:
                    if name not in compute_score_fns:
                        logger.warning(f"Unknown reward_fn: {name}, skipping")
                        continue
                    # Fail loudly if llm_score requires raw_prompt but it's None
                    if name == "llm_score" and raw_prompt is None:
                        raise ValueError(
                            f"[ToolRewardManager] raw_prompt is None for sample {i} but llm_score requires it. "
                            f"This indicates raw_prompt was lost in the pipeline. "
                            f"Check that agent_loop.py preserves raw_prompt in extra_fields."
                        )
                    s = compute_score_fns[name](valid_response_str, ground_truth, **common_kwargs)
                    score[name] = s
                    total_score += s

                if len(reward_fn_names) > 0:
                    score["score"] = total_score

                # Compute unused reward metrics (logged but not used for training)
                for name in unused_reward_fn_names:
                    if name in score:
                        continue
                    if name not in compute_score_fns:
                        logger.warning(f"Unknown unused_reward_fn: {name}, skipping")
                        continue
                    # Fail loudly if llm_score requires raw_prompt but it's None
                    if name == "llm_score" and raw_prompt is None:
                        raise ValueError(
                            f"[ToolRewardManager] raw_prompt is None for sample {i} but llm_score requires it. "
                            f"This indicates raw_prompt was lost in the pipeline. "
                            f"Check that agent_loop.py preserves raw_prompt in extra_fields."
                        )
                    s = compute_score_fns[name](valid_response_str, ground_truth, **common_kwargs)
                    score[name] = s

                # Log number of turns if enabled
                if self.log_num_round:
                    score["num_round"] = valid_response_str.count("<|im_start|>assistant")

                return i, score, valid_response_length

            except Exception as e:
                import traceback

                logger.error(f"Failed to process sample {i}: {type(e).__name__}: {str(e)}")
                logger.debug(traceback.format_exc())
                raise RuntimeError(f"Failed to process sample {i}: {type(e).__name__}: {str(e)}") from None

        # Process all items
        results = []
        for i in range(len(data)):
            result = _process_item(i, data[i])
            results.append(result)

        # Collect all keys
        if results and isinstance(results[0][1], dict):
            all_keys = set()
            for _, score, _ in results:
                all_keys.update(score.keys())

        # Fill reward tensor
        for i, score, valid_response_length in results:
            if isinstance(score, dict):
                if "score" in score:
                    reward = score["score"]
                    if valid_response_length > 0:
                        reward_tensor[i, valid_response_length - 1] = reward
                for key in all_keys:
                    reward_extra_info[key].append(score.get(key, None))
            else:
                reward = score
                if valid_response_length > 0:
                    reward_tensor[i, valid_response_length - 1] = reward

        # Get judge LLM stats and reset counters for next batch
        # Stats are always collected for logging, threshold check is optional
        stats = get_judge_llm_stats(reset=True)
        if stats["judge_llm_total"] > 0:
            # Add batch-level stats to reward_extra_info for wandb logging
            # These are scalars (not per-sample lists) representing batch aggregates
            reward_extra_info["judge_llm_total"] = stats["judge_llm_total"]
            reward_extra_info["judge_llm_successful"] = stats["judge_llm_successful"]
            reward_extra_info["judge_llm_success_rate"] = stats["judge_llm_success_rate"]

            # Check threshold and crash if below (only if threshold is set)
            if self.judge_llm_success_threshold > 0:
                context = "VAL" if self.is_val else "TRAIN"
                success_rate = stats["judge_llm_success_rate"]
                print(
                    f"[JudgeLLM] [{context}] Stats: total={stats['judge_llm_total']}, "
                    f"successful={stats['judge_llm_successful']}, success_rate={success_rate:.2%}",
                    flush=True
                )
                if success_rate < self.judge_llm_success_threshold:
                    error_msg = (
                        f"[JudgeLLM] [{context}] FATAL: Success rate {success_rate:.2%} is below "
                        f"threshold {self.judge_llm_success_threshold:.0%} "
                        f"({stats['judge_llm_successful']}/{stats['judge_llm_total']} calls succeeded). "
                        f"This indicates LLM judge API issues (timeout/rate limit/server error). "
                        f"Crashing to prevent training on incomplete reward data."
                    )
                    print(f"\n{'='*80}\n{error_msg}\n{'='*80}\n", flush=True)
                    raise RuntimeError(error_msg)

        if return_dict:
            logger.debug(f"reward_extra_info keys: {list(reward_extra_info.keys())}")
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
