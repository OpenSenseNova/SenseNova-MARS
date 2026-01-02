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

import json
import os

from ray._private.runtime_env.constants import RAY_JOB_CONFIG_JSON_ENV_VAR

PPO_RAY_RUNTIME_ENV = {
    "env_vars": {
        "TOKENIZERS_PARALLELISM": "true",
        "NCCL_DEBUG": "WARN",
        "VLLM_LOGGING_LEVEL": "WARN",
        "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true",
        # symmetric memory allreduce not work properly in spmd mode
        "VLLM_ALLREDUCE_USE_SYMM_MEM": "0",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        # To prevent hanging or crash during synchronization of weights between actor and rollout
        # in disaggregated mode. See:
        # https://docs.vllm.ai/en/latest/usage/troubleshooting.html?h=nccl_cumem_enable#known-issues
        # https://github.com/vllm-project/vllm/blob/c6b0a7d3ba03ca414be1174e9bd86a97191b7090/vllm/worker/worker_base.py#L445
        "NCCL_CUMEM_ENABLE": "0",
    },
}

# Environment variables to forward from parent process to Ray workers
# These are read from os.environ at runtime and added to runtime_env
ENV_VARS_TO_FORWARD = [
    # Image processing
    "MIN_PIXELS",
    "MAX_PIXELS",
    "IMAGE_FACTOR",
    # External API keys
    "AZURE_OPENAI_API_KEY",
    "OPENAI_API_KEY",
    # Tool server addresses
    "TEXT_SEARCH_ADDRESS",
    "LOCAL_DATABASE_ADDRESS",
    # VLLM configuration
    "VLLM_USE_V1",
    "VLLM_FLASH_ATTN_VERSION",
    "VLLM_ATTENTION_BACKEND",
    "VLLM_BATCH_INVARIANT",
    "VLLM_LOGGING_LEVEL",
    # Logging
    "VERL_LOGGING_LEVEL",
    "PYTHONUNBUFFERED",
    # SGLang configuration
    "SGL_ENABLE_JIT_DEEPGEMM",
    # Deterministic training (like Miles)
    "NVTE_ALLOW_NONDETERMINISTIC_ALGO",
    "CUBLAS_WORKSPACE_CONFIG",
]


def get_ppo_ray_runtime_env():
    """
    A filter function to return the PPO Ray runtime environment.
    To avoid repeat of some environment variables that are already set.
    Also forwards specific env vars (like API keys) from parent process.
    """
    working_dir = (
        json.loads(os.environ.get(RAY_JOB_CONFIG_JSON_ENV_VAR, "{}")).get("runtime_env", {}).get("working_dir", None)
    )

    runtime_env = {
        "env_vars": PPO_RAY_RUNTIME_ENV["env_vars"].copy(),
        **({"working_dir": None} if working_dir is None else {}),
    }
    for key in list(runtime_env["env_vars"].keys()):
        if os.environ.get(key) is not None:
            runtime_env["env_vars"].pop(key, None)

    # Forward specific env vars from parent process to Ray workers
    for key in ENV_VARS_TO_FORWARD:
        value = os.environ.get(key)
        if value is not None:
            runtime_env["env_vars"][key] = value

    return runtime_env
