# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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

import asyncio
import httpx
import json
import logging
import os
import re
from typing import List, Dict, Optional, Any
from openai import AsyncOpenAI, AsyncAzureOpenAI

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Use a dictionary to hold clients for different proxy configs
_global_http_clients: Dict[Optional[str], httpx.AsyncClient] = {}
_client_lock = asyncio.Lock()

async def get_shared_http_client(proxy_url: str = None) -> httpx.AsyncClient:
    """Get or create a shared HTTP client based on the proxy URL."""
    async with _client_lock:
        # Check if a client for this proxy is missing or closed
        if proxy_url not in _global_http_clients or _global_http_clients[proxy_url].is_closed:
            logger.info(f"Creating new shared httpx.AsyncClient for proxy: {proxy_url or 'None'}")
            _global_http_clients[proxy_url] = httpx.AsyncClient(
                proxy=proxy_url,
                limits=httpx.Limits(
                    max_keepalive_connections=20,
                    max_connections=50,
                    keepalive_expiry=60.0
                ),
                timeout=httpx.Timeout(30.0)
            )
        return _global_http_clients[proxy_url]

async def cleanup_shared_resources():
    """Clean up all shared HTTP clients on shutdown."""
    async with _client_lock:
        logger.info("Starting cleanup of shared HTTP clients...")
        for proxy_url, client in list(_global_http_clients.items()):
            if client and not client.is_closed:
                logger.info(f"Closing shared httpx.AsyncClient for proxy: {proxy_url or 'None'}")
                await client.aclose()
        _global_http_clients.clear()
        logger.info("Cleanup of shared HTTP clients complete.")

def _shutdown_handler():
    """Simplified shutdown handler that avoids async deadlocks"""
    try:
        # Simple sync cleanup to avoid event loop issues - more robust approach
        for proxy_url, client in list(_global_http_clients.items()):
            if client and not client.is_closed:
                try:
                    # Try multiple sync cleanup approaches
                    # 1. Close transport directly if available
                    if hasattr(client, '_transport') and client._transport:
                        if hasattr(client._transport, 'close'):
                            client._transport.close()
                        elif hasattr(client._transport, '_pool') and client._transport._pool:
                            # Close connection pool directly
                            if hasattr(client._transport._pool, 'close'):
                                client._transport._pool.close()
                    
                    # 2. Try connector cleanup
                    if hasattr(client, '_connector') and client._connector:
                        if hasattr(client._connector, 'close'):
                            # Note: Don't call async close() - just force cleanup
                            try:
                                client._connector._closed = True
                            except Exception:
                                pass
                                
                    logger.debug(f"Sync cleanup completed for proxy: {proxy_url or 'None'}")
                    
                except Exception as cleanup_error:
                    logger.debug(f"Partial cleanup error for proxy {proxy_url}: {cleanup_error}")
                    # Continue with other clients even if one fails
        
        _global_http_clients.clear()
        logger.debug("All HTTP clients cleared from registry")
        
    except Exception as e:
        logger.warning(f"Error during shutdown cleanup: {e}")

# Register improved cleanup handler
import atexit
atexit.register(_shutdown_handler)


class LLMGenerator:
    """enhanced LLM generator, support multiple models and APIs"""

    def __init__(self, config_path: str = './config.json', max_completion_token: int = 12288, max_try_times: int = 3, progress_tracker=None):
        # 加载配置
        config_data = self._load_config(config_path)
        self.api_configs = config_data

        # Use config values if available, otherwise use defaults
        llm_config = config_data.get('llm', {})
        self.max_completion_token = llm_config.get('max_completion_token', max_completion_token)
        self.max_try_times = llm_config.get('max_try_times', max_try_times)
        self.acc_p_tk = 0
        self.acc_c_tk = 0
        self.progress_tracker = progress_tracker
        self.client = None
        self.model = None

        # Track current client config to avoid unnecessary recreation
        self._current_model_config = None
        self._client_lock = asyncio.Lock()
    
    def _load_config(self, config_path: str) -> Dict:
        """Loads API configuration from a JSON file."""
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as rf:
                    return json.load(rf)
            else:
                # 默认配置
                return {
                    'gpt-3.5-turbo': {
                        'model_name': 'gpt-3.5-turbo',
                        'api_key_var': 'OPENAI_API_KEY',
                        'base_url': 'https://api.openai.com/v1',
                        'provider': 'openai'
                    }
                }
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return {}

    async def _get_or_create_http_client(self, proxy_url: str = None) -> httpx.AsyncClient:
        """Use shared HTTP client instead of per-instance client"""
        return await get_shared_http_client(proxy_url)

    async def close(self):
        """Cleanup method - shared client persists across instances"""
        # Don't close shared client - it's managed globally
        pass
    
    async def create_model(self, model: str = 'gpt-3.5-turbo'):
        """create model client with concurrency protection"""
        async with self._client_lock:
            # Only recreate if model config changed
            if self._current_model_config == model and self.client is not None:
                return
                
            if model not in self.api_configs:
                raise ValueError(f"Model {model} not found in config")
            
            config = self.api_configs[model]
            self.model = config['model_name']
            api_key = os.environ.get(config['api_key_var'])
            provider = config.get('provider', 'openai')
            proxy_url = config.get('proxy_url')
            
            if not api_key:
                # For VLLM/local deployments that don't require authentication,
                # use "EMPTY" as placeholder to avoid "Illegal header value" error
                # in newer OpenAI client versions
                api_key = "EMPTY"
                logger.warning(f"API key for {model} not found in environment variables, using 'EMPTY' as placeholder")
            
            if provider == 'azure':
                azure_endpoint = config.get('azure_endpoint')
                api_version = config.get('api_version', '2025-01-01-preview')
                
                if not azure_endpoint:
                    raise ValueError(f"Azure endpoint for {model} not found in config")
                
                if proxy_url:
                    http_client = await self._get_or_create_http_client(proxy_url)
                    self.client = AsyncAzureOpenAI(
                        api_version=api_version,
                        azure_endpoint=azure_endpoint,
                        api_key=api_key,
                        http_client=http_client
                    )
                else:
                    self.client = AsyncAzureOpenAI(
                        api_version=api_version,
                        azure_endpoint=azure_endpoint,
                        api_key=api_key
                    )
            else:
                base_url_var = config.get('base_url_var')
                base_url = os.environ.get(base_url_var) if base_url_var else None
                if not base_url:
                    raise ValueError(f"Base URL for {model} not found in environment variables")
                
                if proxy_url:
                    http_client = await self._get_or_create_http_client(proxy_url)
                    self.client = AsyncOpenAI(
                        api_key=api_key,
                        base_url=base_url,
                        http_client=http_client
                    )
                else:
                    self.client = AsyncOpenAI(
                        base_url=base_url,
                        api_key=api_key
                    )
            
            self._current_model_config = model
    
    async def calculate_tokens(self, response) -> None:
        try:
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            
            self.acc_p_tk += prompt_tokens
            self.acc_c_tk += completion_tokens
            
            logger.info(f"Prompt tokens: {prompt_tokens}/{self.acc_p_tk}, Completion tokens: {completion_tokens}/{self.acc_c_tk}")
        except Exception as e:
            logger.warning(f"Failed to calculate tokens: {e}")
    
    async def generate(self, messages: List[Dict], model: str = 'gpt-3.5-turbo', timeout: int = 180, enable_thinking: bool = False) -> str:
        # Safely create/recreate client if model changed or client doesn't exist
        if self._current_model_config != model or self.client is None:
            await self.create_model(model)
        try_times = 0

        # Track LLM start
        if self.progress_tracker:
            self.progress_tracker.update_llm_progress(active_delta=1)

        while try_times < self.max_try_times:
            try:
                logger.info(f"LLM generation attempt {try_times + 1}/{self.max_try_times} - Model: {model}, Timeout: {timeout}s, Thinking: {enable_thinking}")

                # Use appropriate token parameter based on model
                token_params = {}
                if any(model_name in self.model.lower() for model_name in ['gpt-4o-mini', 'gpt-4o', 'qwen']):
                    token_params['max_tokens'] = self.max_completion_token
                else:
                    token_params['max_completion_tokens'] = self.max_completion_token

                # For Qwen3 models via vLLM, use extra_body to control thinking mode
                # This is safely ignored by non-Qwen3 models and non-vLLM backends (Azure OpenAI, OpenAI, etc.)
                if 'qwen3' in self.model.lower():
                    token_params['extra_body'] = {
                        "chat_template_kwargs": {"enable_thinking": enable_thinking}
                    }

                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    timeout=timeout,
                    **token_params
                )

                # Track success and tokens
                if self.progress_tracker:
                    prompt_tokens = response.usage.prompt_tokens if response.usage else 0
                    completion_tokens = response.usage.completion_tokens if response.usage else 0
                    self.progress_tracker.update_llm_progress(
                        active_delta=-1,
                        completed_delta=1,
                        prompt_tokens_delta=prompt_tokens,
                        completion_tokens_delta=completion_tokens,
                        retries_delta=try_times  # Track total retries for this successful call
                    )

                await self.calculate_tokens(response)
                content = response.choices[0].message.content
                logger.info(f"LLM generation successful on attempt {try_times + 1} - Response length: {len(content) if content else 0} chars")
                return content

            except Exception as e:
                logger.error(f"LLM generation error (attempt {try_times + 1}/{self.max_try_times}): {type(e).__name__}: {e}")
                logger.error(f"LLM error details - Model: {model}, Messages count: {len(messages)}")
                if messages:
                    total_chars = sum(len(str(msg.get('content', ''))) for msg in messages)
                    logger.error(f"LLM error details - Total content chars: {total_chars}")
                import traceback
                logger.error(f"LLM error traceback: {traceback.format_exc()}")
                try_times += 1
                if try_times < self.max_try_times:
                    logger.warning(f"Retrying LLM generation in 1s (attempt {try_times + 1}/{self.max_try_times})")
                    await asyncio.sleep(1)

        # Track failure after all retries exhausted
        if self.progress_tracker:
            self.progress_tracker.update_llm_progress(
                active_delta=-1,
                failed_delta=1,
                retries_delta=try_times  # Track total retries for this failed call
            )

        return ""

    def _clean_summary(self, summary: str) -> str:
        """Remove <think>...</think> blocks and strip the result"""
        if not summary:
            return summary

        # Check if think blocks exist before cleaning
        think_pattern = re.compile(r'<think>.*?</think>', re.DOTALL | re.IGNORECASE)
        think_blocks = think_pattern.findall(summary)

        if think_blocks:
            total_think_chars = sum(len(block) for block in think_blocks)
            logger.info(f"[THINKING] ✓ Detected {len(think_blocks)} <think> block(s), {total_think_chars} chars of reasoning content - removing from final output")
            # Log a sample of the first thinking block for verification
            if len(think_blocks[0]) > 100:
                logger.debug(f"[THINKING] Sample from first block: {think_blocks[0][:100]}...")
        else:
            logger.debug(f"[THINKING] ✗ No <think> blocks detected in response")

        # Remove complete <think>...</think> blocks (case insensitive, multiline)
        cleaned = re.sub(r'<think>.*?</think>', '', summary, flags=re.DOTALL | re.IGNORECASE)

        return cleaned.strip()

    async def generate_summary(self, query: str, content: str, model: str = 'gpt-3.5-turbo', prompt_type: str = 'gpt4o', content_limit: int = 100000, enable_thinking: bool = False) -> str:
        logger.info(f"[LLM] Generating summary with: model={model}, prompt={prompt_type}, content_limit={content_limit}, thinking={enable_thinking}")

        try:
            # Select prompts based on prompt_type
            if prompt_type == 'mmsearch_r1':
                logger.info(f"[LLM] Using mmsearch_r1 prompt: 5-sentence summary style")
                system_prompt = """You are a helpful assistant. Your task is to summarize the main content of the given web page in no more than five sentences. Your summary should cover the overall key points of the page, not just parts related to the user's question.

If any part of the content is helpful for answering the user's question, be sure to include it clearly in the summary. Do not ignore relevant information, but also make sure the general structure and main ideas of the page are preserved. Your summary should be concise, factual, and informative."""

                user_prompt = f"""Webpage Content (first {content_limit} characters) is: {{content}}
Question: {{query}}"""
            else:  # default to gpt4o
                logger.info(f"[LLM] Using gpt4o prompt: expert researcher style")
                system_prompt = """You are an expert researcher. Follow these instructions when responding:
  - You may be asked to research subjects that is after your knowledge cutoff, assume the user is right when presented with news.
  - The user is a highly experienced analyst, no need to simplify it, be as detailed as possible and make sure your response is correct.
  - Be highly organized.
  - Suggest solutions that I didn't think about.
  - Be proactive and anticipate my needs.
  - Treat me as an expert in all subject matter.
  - Mistakes erode my trust, so be accurate and thorough.
  - Provide detailed explanations, I'm comfortable with lots of detail.
  - Value good arguments over authorities, the source is irrelevant.
  - Consider new technologies and contrarian ideas, not just the conventional wisdom.
  - You may use high levels of speculation or prediction, just flag it for me."""

                user_prompt = """Given raw webpage contents: <contents>{content}</contents>, compress these contents into a **maximum 2K-token contents** adhering to:
1. Preserve critical information, logical flow, and essential data points
2. Prioritize content relevance to the research query:
   <query>{query}</query>
3. **Adjust length dynamically**:
   - If original content < 2K tokens, maintain original token count ±10%
   - If original content ≥ 2K tokens, compress to ~2K tokens
4. Format output in clean Markdown without decorative elements \n
**Prohibited**:
- Adding content beyond source material
- Truncating mid-sentence to meet token limits"""
            
            formatted_user_prompt = user_prompt.format(
                content=content[:content_limit],
                query=query
            )

            messages = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': formatted_user_prompt}
            ]

            summary = await self.generate(messages, model, enable_thinking=enable_thinking)
            return self._clean_summary(summary)

        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return f"summary error: {str(e)}"
    
    async def translate_query(self, query: str, target_language: str = 'zh', model: str = 'gpt-3.5-turbo') -> str:
        """Translate the query to the target language"""
        try:
            system_prompt = f"""please translate the following query to {target_language}, requirements:
            1. keep the original intent and meaning of the query
            2. use accurate professional terms
            3. ensure correct grammar
            4. do not add additional explanations

            please return the translation directly: """ 

            user_prompt = f"query: {query}"

            messages = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
            ]
            
            translation = await self.generate(messages, model)
            return translation.strip()
            
        except Exception as e:
            logger.error(f"Error translating query: {e}")
            return query 
