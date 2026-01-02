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
Tool Reward Loop Manager for VLM multi-turn tool training.

This is an async adapter for the experimental reward loop framework.
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

import json
import logging
import os
import random
import re
from collections import defaultdict

import openai
from omegaconf import DictConfig
from openai import AzureOpenAI
from transformers import AutoTokenizer

from verl import DataProto
from verl.experimental.reward.reward_loop import register
from verl.experimental.reward.reward_loop.base import RewardLoopManagerBase
from verl.utils.reward_score.tool import (
    compute_score_fns,
    configure_judge_llm_semaphore,
    convert_hf_messages_to_openai,
)


def _validate_format_from_response(response_str: str) -> bool:
    """
    Validate format from decoded response string for format_score reward.
    Checks:
    1. Has <answer>...</answer> with non-empty content in final turn
    2. Tool calls (if any) are valid JSON
    """
    # Split into assistant turns (don't slice [1:] - first turn doesn't start with the token)
    assistant_turns = [t for t in response_str.split("<|im_start|>assistant") if t.strip()]

    if not assistant_turns:
        return False

    # Get last turn content (up to <|im_end|>)
    last_turn = assistant_turns[-1]
    end_pos = last_turn.find('<|im_end|>')
    if end_pos != -1:
        last_turn = last_turn[:end_pos]

    # Check for <answer> tag with non-empty content (use last match like scorer does)
    answer_matches = list(re.finditer(r'<answer>(.*?)</answer>', last_turn, re.DOTALL))
    if not answer_matches:
        return False
    answer_content = answer_matches[-1].group(1).strip()
    if not answer_content:
        return False

    # Check tool calls in non-final turns are valid JSON
    for turn in assistant_turns[:-1]:
        end_pos = turn.find('<|im_end|>')
        if end_pos != -1:
            turn = turn[:end_pos]

        tool_calls = re.findall(r'<tool_call>(.*?)</tool_call>', turn, re.DOTALL)
        for tc in tool_calls:
            try:
                json.loads(tc.strip())
            except json.JSONDecodeError:
                return False

    return True

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("tool")
class ToolRewardLoopManager(RewardLoopManagerBase):
    """Async reward loop manager for VLM multi-turn tool training.

    Reads reward_fn and unused_reward_fn from dataset configuration to compute
    training rewards and evaluation metrics.
    """

    def __init__(
        self,
        config: DictConfig,
        tokenizer: AutoTokenizer,
        compute_score=None,
        reward_router_address=None,
        reward_model_tokenizer=None,
    ):
        super().__init__(config, tokenizer)
        # Note: reward_router_address and reward_model_tokenizer are unused
        # but required by RewardManagerWorker interface
        self.compute_score = compute_score_fns

        # Get reward config from config
        reward_config = getattr(config, "reward_model", config)
        reward_kwargs = getattr(reward_config, "reward_kwargs", {})

        # Get val_reward_kwargs, fall back to reward_kwargs if not specified (same as main_ppo.py)
        val_reward_kwargs = getattr(reward_config, "val_reward_kwargs", None)
        if val_reward_kwargs is None:
            val_reward_kwargs = reward_kwargs

        # Convert OmegaConf to dict if needed
        if hasattr(reward_kwargs, "items"):
            reward_kwargs = dict(reward_kwargs)
        if hasattr(val_reward_kwargs, "items"):
            val_reward_kwargs = dict(val_reward_kwargs)

        self.reward_fn_key = reward_kwargs.get("reward_fn_key", "data_source")
        self.max_turns = reward_kwargs.get("max_turns", 10)
        self.is_val = reward_kwargs.get("is_val", False)
        self.log_num_round = reward_kwargs.get("log_num_round", False)
        self.format_score = reward_kwargs.get("format_score", 0.5)
        self.val_format_score = val_reward_kwargs.get("format_score", self.format_score)

        # Initialize TRAINING LLM judge clients
        self.train_clients, self.train_llm_judge_config = self._init_llm_judge_clients(reward_kwargs, "TRAIN")

        # Initialize VALIDATION LLM judge clients
        self.val_clients, self.val_llm_judge_config = self._init_llm_judge_clients(val_reward_kwargs, "VAL")

        # LLM judge success rate threshold (0.0-1.0, crash if below)
        # Set to 0 to disable the check
        # Note: For async reward loop, check should happen at batch level (caller's responsibility)
        self.judge_llm_success_threshold = reward_kwargs.get("judge_llm_success_threshold", 0.95)

        # Configure semaphore for rate limiting LLM judge API calls
        # Use the max of train and val concurrency limits
        train_concurrency = reward_kwargs.get("llm_judge_concurrency_limit", 0)
        val_concurrency = val_reward_kwargs.get("llm_judge_concurrency_limit", 0)
        llm_judge_concurrency_limit = max(train_concurrency, val_concurrency)
        if llm_judge_concurrency_limit > 0:
            configure_judge_llm_semaphore(llm_judge_concurrency_limit)
            logger.info(f"Configured LLM judge semaphore with limit={llm_judge_concurrency_limit}")

    def _init_llm_judge_clients(self, kwargs: dict, mode: str):
        """Initialize LLM judge clients for training or validation.

        Args:
            kwargs: reward_kwargs or val_reward_kwargs dict
            mode: "TRAIN" or "VAL" for logging

        Returns:
            tuple: (clients list, llm_judge_config dict or None)
        """
        llm_judge_model = kwargs.get("llm_judge_model")
        llm_judge_urls = kwargs.get("llm_judge_urls", [])
        llm_judge_temperature = kwargs.get("llm_judge_temperature", 0.0)
        llm_judge_max_tokens = kwargs.get("llm_judge_max_tokens", 8192)
        llm_judge_timeout = kwargs.get("llm_judge_timeout", 120)
        llm_judge_is_vision_model = kwargs.get("llm_judge_is_vision_model", True)
        llm_judge_enable_thinking = kwargs.get("llm_judge_enable_thinking", False)

        logger.info(f"[{mode}] llm_judge_model={llm_judge_model}")
        print(f"[ToolRewardLoopManager] [{mode}] llm_judge_model={llm_judge_model}")

        clients = []
        llm_judge_config = None

        if llm_judge_model:
            is_azure = "gpt-4o" in llm_judge_model.lower()

            if is_azure:
                # GPT models -> Azure OpenAI
                llm_judge_api_key = kwargs.get("llm_judge_api_key", os.getenv("AZURE_OPENAI_API_KEY"))
                azure_endpoint = kwargs.get("llm_judge_azure_endpoint", "https://duomotai.openai.azure.com/")
                api_version = kwargs.get("llm_judge_azure_api_version", "2025-01-01-preview")

                clients = [
                    AzureOpenAI(
                        api_key=llm_judge_api_key,
                        azure_endpoint=azure_endpoint,
                        api_version=api_version,
                        max_retries=5,
                    )
                ]
                logger.info(f"[{mode}] Using Azure OpenAI for {llm_judge_model}")
                print(f"[ToolRewardLoopManager] [{mode}] Using Azure OpenAI for {llm_judge_model}")

            else:
                # Non-GPT models -> vLLM (requires URLs)
                if not llm_judge_urls:
                    raise ValueError(
                        f"[{mode}] Model '{llm_judge_model}' requires llm_judge_urls.\n"
                        f"Example: +reward_model.reward_kwargs.llm_judge_urls=['10.119.27.50:8181']"
                    )

                llm_judge_api_key = kwargs.get("llm_judge_api_key", "token-abc123")
                clients = [
                    openai.OpenAI(api_key=llm_judge_api_key, base_url=f"http://{url}/v1", max_retries=5)
                    for url in llm_judge_urls
                ]
                logger.info(f"[{mode}] Using vLLM for {llm_judge_model} with {len(llm_judge_urls)} server(s)")
                print(f"[ToolRewardLoopManager] [{mode}] Using vLLM at {llm_judge_urls}")

            llm_judge_config = {
                "model": llm_judge_model,
                "temperature": llm_judge_temperature,
                "max_tokens": llm_judge_max_tokens,
                "timeout": llm_judge_timeout,
                "is_vision_model": llm_judge_is_vision_model,
                "enable_thinking": llm_judge_enable_thinking,
            }

        return clients, llm_judge_config

    async def run_single(self, data: DataProto) -> dict:
        """Process a single data item asynchronously.

        Args:
            data: DataProto containing a single batch item

        Returns:
            dict with reward_score and reward_extra_info
        """
        assert len(data) == 1, "Only support single data item"
        data_item = data[0]

        try:
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]

            response_ids = data_item.batch["responses"]
            response_length = response_ids.shape[-1]
            valid_response_length = data_item.batch["attention_mask"][-response_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # Decode response asynchronously
            valid_response_str = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.decode(valid_response_ids)
            )

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            reward_fn_names = data_item.non_tensor_batch["reward_model"]["reward_fn"]
            unused_reward_fn_names = data_item.non_tensor_batch["reward_model"]["unused_reward_fn"]


            raw_prompt = data_item.non_tensor_batch.get("raw_prompt")
            multi_modal_inputs = data_item.non_tensor_batch.get("multi_modal_inputs", None)
            multi_modal_data = data_item.non_tensor_batch.get("multi_modal_data", None)

            # Convert HF format messages to OpenAI format for LLM judge API
            # HF format: {"type": "image"} with separate PIL images
            # OpenAI format: {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
            if raw_prompt is not None and multi_modal_data is not None:
                images = multi_modal_data.get("image", [])
                if images:
                    raw_prompt = convert_hf_messages_to_openai(raw_prompt, images)

            score = {}

            # Determine if this is a validation batch and select appropriate clients/config
            # After DataProto[0] indexing, __is_validate__ is np.bool_ scalar
            is_validate = bool(data_item.non_tensor_batch.get("__is_validate__", False))
            if is_validate:
                clients = self.val_clients
                llm_config = self.val_llm_judge_config
                format_score = self.val_format_score
            else:
                clients = self.train_clients
                llm_config = self.train_llm_judge_config
                format_score = self.format_score

            # Common kwargs for scoring functions
            common_kwargs = dict(
                clients=clients,
                llm_config=llm_config,
                raw_prompt=raw_prompt,
                multi_modal_inputs=multi_modal_inputs,
                multi_modal_data=multi_modal_data,
                max_turns=self.max_turns,
                eos_token=self.tokenizer.eos_token,
                data_non_tensor_batch=data_item.non_tensor_batch,
                is_val=is_validate,
                format_score=format_score,
            )

            # Compute is_valid_format for format_score reward
            is_valid_format = _validate_format_from_response(valid_response_str)
            common_kwargs['is_valid_format'] = is_valid_format

            # Compute reward scores
            total_score = 0.0
            for name in reward_fn_names:
                if name not in compute_score_fns:
                    logger.warning(f"Unknown reward_fn: {name}, skipping")
                    continue
                # Run scoring in executor to avoid blocking
                s = await self.loop.run_in_executor(
                    None,
                    lambda n=name: compute_score_fns[n](valid_response_str, ground_truth, **common_kwargs)
                )
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
                # Run scoring in executor to avoid blocking
                s = await self.loop.run_in_executor(
                    None,
                    lambda n=name: compute_score_fns[n](valid_response_str, ground_truth, **common_kwargs)
                )
                score[name] = s

            # Log number of turns if enabled
            if self.log_num_round:
                score["num_round"] = valid_response_str.count("<|im_start|>assistant")

            # Build reward_extra_info
            reward_extra_info = {}
            for key, value in score.items():
                reward_extra_info[key] = value

            # Get final reward
            reward = score.get("score", 0.0)

            # Note: Judge LLM stats are aggregated at batch level in AgentLoopManager.generate_sequences
            # Each worker tracks its own stats, which are collected and aggregated after batch completion

            return {"reward_score": reward, "reward_extra_info": reward_extra_info}

        except RuntimeError:
            # Re-raise RuntimeError (e.g., judge LLM failure) to crash training
            raise
        except Exception as e:
            import traceback
            logger.error(f"Failed to process sample: {type(e).__name__}: {str(e)}")
            logger.debug(traceback.format_exc())
            # Return zero reward on error
            return {"reward_score": 0.0, "reward_extra_info": {"error": str(e)}}
