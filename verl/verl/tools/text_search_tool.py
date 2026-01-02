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
Text Search Tool for VLM multi-turn training.

Supports two backends:
- local_retrieval: Local database retrieval with LLM summarization
- google_serper: Google Serper API search with LLM summarization

Both backends communicate with text_search_server via HTTP.
"""

import asyncio
import logging
import os
import random
from typing import Any, Optional
from uuid import uuid4

import aiohttp

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class TextSearchTool(BaseTool):
    """
    A tool for text search with LLM-powered summarization.

    Supports two modes:
    - local_retrieval: Search local knowledge base
    - google_serper: Search via Google Serper API

    Both modes use HTTP calls to text_search_server for actual search and summarization.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """
        Initialize TextSearchTool.

        Config options:
            text_search_type: "local_retrieval" or "google_serper"
            text_search_address: Address of text_search_server (e.g., "10.119.27.135:8000")
            local_database_address: Address of local database (for local_retrieval)
            text_search_topk: Number of results to return (default: 3)
            text_search_llm_model: LLM model for summarization (default: "gpt-4o-mini")
            text_search_prompt_type: Prompt type (default: "gpt4o")
            text_search_use_jina: Use Jina for page fetching (default: False)
            text_search_enable_thinking: Enable thinking mode (default: False)
            timeout: Request timeout in seconds (default: 300)
            max_attempts: Max retry attempts (default: 3)
        """
        super().__init__(config, tool_schema)
        self._instance_dict = {}

        # Required config - support both old (web_search_*) and new (text_search_*) config keys
        self.text_search_type = config.get("text_search_type") or config.get("web_search_type")
        self.text_search_address = config.get("text_search_address") or config.get("web_search_address")

        if not self.text_search_type:
            raise ValueError("text_search_type is required (local_retrieval or google_serper)")
        if not self.text_search_address:
            raise ValueError("text_search_address is required")

        # Optional config - support both old and new config keys
        self.local_database_address = config.get("local_database_address", "")
        self.text_search_topk = config.get("text_search_topk") or config.get("web_search_topk", 3)
        self.text_search_llm_model = config.get("text_search_llm_model") or config.get("web_search_llm_model", "gpt-4o-mini")
        self.text_search_prompt_type = config.get("text_search_prompt_type") or config.get("web_search_prompt_type", "gpt4o")
        self.text_search_use_jina = config.get("text_search_use_jina") or config.get("web_search_use_jina", False)
        self.text_search_enable_thinking = config.get("text_search_enable_thinking") or config.get("web_search_enable_thinking", False)
        self.timeout = config.get("timeout", 300)
        self.max_attempts = config.get("max_attempts", 3)

        # Rate limiting
        self.max_concurrent = config.get("max_concurrent", 10)
        self._semaphore = asyncio.Semaphore(self.max_concurrent)

        logger.info(
            f"Initialized TextSearchTool: type={self.text_search_type}, "
            f"address={self.text_search_address}, topk={self.text_search_topk}"
        )

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        """Create a tool instance."""
        if instance_id is None:
            instance_id = str(uuid4())

        self._instance_dict[instance_id] = {
            "queries": [],
            "results": [],
        }
        return instance_id, ToolResponse()

    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs
    ) -> tuple[ToolResponse, float, dict]:
        """
        Execute text search.

        Args:
            instance_id: Instance ID from create()
            parameters: Tool parameters including:
                - query: Search query string
            **kwargs: Additional kwargs

        Returns:
            Tuple of (ToolResponse, reward, info_dict)
        """
        query = parameters.get("query")

        if not query or not isinstance(query, str):
            return (
                ToolResponse(text="Error: query parameter is required and must be a string."),
                -0.05,
                {"success": False, "error": "invalid_query"},
            )

        # Perform search with retry
        result = None
        last_error = None

        for attempt in range(self.max_attempts):
            try:
                async with self._semaphore:
                    result = await self._search(query)
                break
            except Exception as e:
                last_error = e
                logger.warning(f"Text search attempt {attempt + 1}/{self.max_attempts} failed: {e}")
                if attempt < self.max_attempts - 1:
                    await asyncio.sleep(1.0 * (attempt + 1))  # Exponential backoff

        if result is None:
            error_msg = f"Text search failed after {self.max_attempts} attempts: {last_error}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        # Store result
        if instance_id in self._instance_dict:
            self._instance_dict[instance_id]["queries"].append(query)
            self._instance_dict[instance_id]["results"].append(result)

        # Note: <tool_response> tags are added by the chat template, not here
        return (
            ToolResponse(text=result),
            0.0,
            {"success": True, "query": query},
        )

    async def _search(self, query: str) -> str:
        """
        Perform the actual search.

        Args:
            query: Search query

        Returns:
            Search results as formatted string
        """
        # Sample 1/64 (~1.5%) of logs to reduce spam while maintaining observability
        do_print = random.randint(1, 64) == 1

        if self.text_search_type == "google_serper":
            if do_print:
                print(f"[TextSearchTool] mode=google_serper, address={self.text_search_address}, query={query[:50]}...", flush=True)
            return await self._search_google_serper(query)
        elif self.text_search_type == "local_retrieval":
            if do_print:
                print(f"[TextSearchTool] mode=local, address={self.text_search_address}, db={self.local_database_address}, query={query[:50]}...", flush=True)
            return await self._search_local_retrieval(query)
        else:
            raise ValueError(f"Unknown text_search_type: {self.text_search_type}")

    async def _search_google_serper(self, query: str) -> str:
        """
        Search via Google Serper with LLM summarization.

        HTTP call to text_search_server's /search endpoint.
        """
        payload = {
            "query": query,
            "top_k": self.text_search_topk,
            "retrieval_mode": "google_serper",
            "llm_model": self.text_search_llm_model,
            "prompt_type": self.text_search_prompt_type,
            "use_jina": self.text_search_use_jina,
            "enable_thinking": self.text_search_enable_thinking,
        }

        timeout = aiohttp.ClientTimeout(total=self.timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                f"http://{self.text_search_address}/search",
                json=payload,
            ) as response:
                response.raise_for_status()
                return await response.text()

    async def _search_local_retrieval(self, query: str) -> str:
        """
        Search local knowledge base with LLM summarization.

        HTTP call to text_search_server's /search endpoint.
        """
        payload = {
            "query": query,
            "top_k": self.text_search_topk,
            "retrieval_mode": "local",
            "local_database_address": self.local_database_address,
            "llm_model": self.text_search_llm_model,
            "prompt_type": self.text_search_prompt_type,
            "enable_thinking": self.text_search_enable_thinking,
        }

        timeout = aiohttp.ClientTimeout(total=self.timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                f"http://{self.text_search_address}/search",
                json=payload,
            ) as response:
                response.raise_for_status()
                return await response.text()

    async def release(self, instance_id: str, **kwargs) -> None:
        """Release the tool instance."""
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
