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
import hashlib
import json
import logging
import re
import time
from datetime import datetime
from typing import Dict, Any, Tuple, Optional

from config import RetrievalMode, SearchConfig
from search_providers import SearchProvider, GoogleSerperProvider, LocalRetrievalProvider
from content_processor import ContentProcessor
from llm_generator import LLMGenerator
from cache import SQLiteCache

logger = logging.getLogger(__name__)


def _truncate_query(query: str, max_length: int = 60) -> str:
    """Truncate query text with ellipsis if too long"""
    if len(query) <= max_length:
        return query
    return query[:max_length-3] + "..."


class SearchOrchestrator:
    """Orchestrates the entire search, content processing, and summarization pipeline"""
    
    def __init__(self, providers: Dict[str, SearchProvider], content_processor: ContentProcessor,
                 cache: SQLiteCache, llm_generator: LLMGenerator, config: SearchConfig,
                 search_semaphore: asyncio.Semaphore, llm_semaphore: asyncio.Semaphore):
        self.providers = providers
        self.content_processor = content_processor
        self.cache = cache
        self.llm_generator = llm_generator
        self.config = config
        self.search_semaphore = search_semaphore
        self.llm_semaphore = llm_semaphore

    def _make_cache_key(self, query: str, top_k: int, llm_model: str, prompt_type: str, use_jina: bool, enable_thinking: bool) -> str:
        """Generate strict cache key including all parameters that affect output"""
        # Normalize query exactly like memory_bank_tool._normalize_query()
        normalized = re.sub(r'\s+', ' ', query.strip().lower())

        # Build composite key with all result-affecting parameters
        parts = [
            f"q={normalized}",
            f"k={top_k}",
            f"model={llm_model}",
            f"prompt={prompt_type}",
            f"jina={int(bool(use_jina))}",
            f"think={int(bool(enable_thinking))}"
        ]

        return hashlib.md5("|".join(parts).encode("utf-8")).hexdigest()

    async def process_query(self, parameters: str) -> Tuple[str, float, Dict[str, Any]]:
        """Main orchestration method that coordinates the entire search pipeline"""
        async with self.search_semaphore:  # Limit concurrent searches
            try:
                # Parse and validate parameters
                parsed_params = self._parse_and_validate_parameters(parameters)
                if isinstance(parsed_params, tuple):  # Error case
                    return parsed_params
                
                query = parsed_params['query']
                dynamic_top_k = parsed_params['dynamic_top_k']
                retrieval_mode = parsed_params['retrieval_mode']
                local_database_address = parsed_params.get('local_database_address', '')
                llm_model = parsed_params['llm_model']
                prompt_type = parsed_params['prompt_type']
                use_jina = parsed_params['use_jina']
                enable_thinking = parsed_params['enable_thinking']

                logger.info(f"[{retrieval_mode.value.upper()}] Processing query: {_truncate_query(query)}")

                # Step 1: Check cache (skip for local retrieval)
                if retrieval_mode != RetrievalMode.LOCAL:
                    cache_key = self._make_cache_key(query, dynamic_top_k, llm_model, prompt_type, use_jina, enable_thinking)
                    cached = await self.cache.get(cache_key)
                    if cached:
                        cached_summary = cached["summaries"]
                        logger.debug(f"[CACHE] ✓ Using cached result for: {_truncate_query(query)}")
                        return f"Found cached summary for query: {query}\n{cached_summary}", 0.0, {"cached": True}
                else:
                    logger.info(f"[LOCAL] Bypassing ALL cache for local retrieval: {_truncate_query(query)}")

                # Step 2: Perform search
                all_search_results, search_results = await self._perform_search(
                    query, dynamic_top_k, retrieval_mode, local_database_address
                )

                if not search_results:
                    # This is a VALID response - the API succeeded but returned 0 results
                    # This should be in training data so the model learns to handle no-results queries
                    logger.info(f"[{retrieval_mode.value.upper()}] No results found for query: {_truncate_query(query)}")
                    return f"No search results found for query: {query}", 0.0, {}

                # Step 3: Process content and generate summaries
                summaries = await self.content_processor.process_search_results(
                    query, search_results, retrieval_mode, llm_model, prompt_type, use_jina, enable_thinking
                )

                # Step 4: Generate final summary
                final_summary, result_metadata = await self._generate_final_summary(
                    query, all_search_results, summaries, llm_model, prompt_type, enable_thinking
                )

                if final_summary.startswith("Error:"):
                    return final_summary, 0.0, {}

                # Step 5: Update cache (skip for local retrieval)
                result_with_summary = {
                    "search_results": all_search_results,
                    "summaries": final_summary,
                    "summary_metadata": result_metadata
                }

                if retrieval_mode != RetrievalMode.LOCAL:
                    cache_key = self._make_cache_key(query, dynamic_top_k, llm_model, prompt_type, use_jina, enable_thinking)
                    await self.cache.set(cache_key, result_with_summary)
                else:
                    logger.info(f"[LOCAL] Skipping cache storage for local retrieval")

                return f"Final summary generated for query: {query}\n{final_summary}", 0.0, {"cached": False}

            except json.JSONDecodeError:
                return "Error: Invalid JSON format for parameters", 0.0, {}
            except RuntimeError as e:
                # Re-raise RuntimeError (search/retrieval failures) to prevent data contamination
                logger.error(f"[TOOL] ✗ Search failure (re-raising): {e}")
                raise
            except Exception as e:
                logger.error(f"[TOOL] ✗ Execution error: {e}")
                return f"Error: An error occurred: {str(e)}", 0.0, {}

    def _parse_and_validate_parameters(self, parameters: str) -> Dict[str, Any]:
        """Parse and validate input parameters"""
        _parameters = json.loads(parameters)
        query = _parameters.get("query", "")
        dynamic_top_k = _parameters.get("top_k", self.config.top_k)

        # Get retrieval mode and local database address - NO DEFAULTS
        retrieval_mode_str = _parameters.get("retrieval_mode")
        local_database_address = _parameters.get("local_database_address", "") or _parameters.get("web_search_address", "")

        # Get dynamic LLM parameters with defaults
        llm_model = _parameters.get("llm_model", "gpt-4o-mini")
        prompt_type = _parameters.get("prompt_type", "gpt4o")
        use_jina = _parameters.get("use_jina", False)
        enable_thinking = _parameters.get("enable_thinking", False)

        # Log parameter parsing for verification
        logger.info(f"[PARAMS] Parsed: model={llm_model}, prompt={prompt_type}, jina={use_jina}, thinking={enable_thinking}")

        if not query:
            return "Error: Query is required", 0.0, {}

        if not retrieval_mode_str:
            return "Error: retrieval_mode is required (must be 'google_serper' or 'local')", 0.0, {}

        try:
            retrieval_mode = RetrievalMode(retrieval_mode_str)
        except ValueError:
            return f"Error: Invalid retrieval_mode '{retrieval_mode_str}' (must be 'google_serper' or 'local')", 0.0, {}

        if retrieval_mode == RetrievalMode.LOCAL and not local_database_address:
            return "Error: local_database_address is required for local retrieval mode", 0.0, {}

        return {
            'query': query,
            'dynamic_top_k': dynamic_top_k,
            'retrieval_mode': retrieval_mode,
            'local_database_address': local_database_address,
            'llm_model': llm_model,
            'prompt_type': prompt_type,
            'use_jina': use_jina,
            'enable_thinking': enable_thinking
        }

    async def _perform_search(self, query: str, top_k: int, retrieval_mode: RetrievalMode, 
                            local_database_address: str = "") -> Tuple[list, list]:
        """Perform search using appropriate provider"""
        if retrieval_mode == RetrievalMode.LOCAL:
            provider = self.providers['local']
            return await provider.search(query, top_k, local_database_address)
        else:
            provider = self.providers['google_serper']
            return await provider.search(query, top_k)

    async def _generate_final_summary(self, query: str, all_search_results: list,
                                    summaries: list, llm_model: str, prompt_type: str, enable_thinking: bool = False) -> Tuple[str, Dict[str, Any]]:
        """Generate final summary using LLM"""
        # Check if all URL summaries failed
        successful_summaries = [s for s in summaries if s is not None and s.strip()]
        
        # Determine summary completeness for tracking
        total_urls = len(summaries)
        successful_urls = len(successful_summaries)
        
        # Collect all content (search results and summaries) - no fallback strategy for now
        search_content = self.content_processor.format_all_content(
            query, all_search_results, summaries
        )
        content_length = len(search_content)
        logger.info(f"[FINAL] Processing final LLM with {successful_urls}/{total_urls} summaries, content length: {content_length} chars")
        
        if successful_urls == total_urls:
            summary_type = "complete"
            completion_reason = "all_urls_successful"
        else:
            summary_type = "partial_mixed"
            completion_reason = f"{successful_urls}_of_{total_urls}_urls_successful"

        # Generate final summary using LLM
        logger.info(f"[FINAL_LLM] Starting final summary generation using model: {llm_model}")
        final_llm_start_time = time.time()

        try:
            # Acquire semaphore for final summary
            await self.llm_semaphore.acquire()

            try:
                final_summary = await self.llm_generator.generate_summary(
                    query, search_content, llm_model, prompt_type, self.config.llm_content_limit, enable_thinking
                )
                final_llm_elapsed = time.time() - final_llm_start_time
                
                if final_summary and not final_summary.startswith("summary error"):
                    logger.info(f"[FINAL_LLM] ✓ Final summary generated in {final_llm_elapsed:.1f}s: {len(final_summary)} chars")
                    # Log final summary content for debugging
                    logger.info(f"[FINAL_LLM] Final summary preview: {final_summary[:500]}{'...' if len(final_summary) > 500 else ''}")
                else:
                    logger.error(f"[FINAL_LLM] ✗ Final summary failed in {final_llm_elapsed:.1f}s: {final_summary}")
                    return f"Error: Failed to generate final summary for query: {query}", {}
            finally:
                self.llm_semaphore.release()
                
        except Exception as llm_error:
            final_llm_elapsed = time.time() - final_llm_start_time
            logger.error(f"[FINAL_LLM] ✗ Final summarization exception in {final_llm_elapsed:.1f}s: {type(llm_error).__name__}: {llm_error}")
            return f"Error: Failed to generate final summary for query: {query}", {}
        
        if final_summary.startswith("summary error"):
            return f"Error: Failed to generate final summary for query: {query}", {}

        result_metadata = {
            "type": summary_type,
            "completion_reason": completion_reason,
            "total_urls": total_urls,
            "successful_urls": successful_urls,
            "is_complete": summary_type == "complete",
            "created_timestamp": datetime.now().isoformat()
        }

        return final_summary, result_metadata

    async def regenerate_summary_from_cache(self, cached_data: dict, original_query: str = None) -> Tuple[str, float, Dict[str, Any]]:
        """
        Regenerate summary using cached search results by visiting pages again and re-running LLM summarization.
        This is useful when previous page visits failed or need to be refreshed.
        """
        async with self.search_semaphore:
            try:
                if not cached_data or "search_results" not in cached_data:
                    return "Error: Invalid cached data provided", 0.0, {}

                search_results = cached_data["search_results"]
                query = original_query or "Regenerated query"

                logger.info(f"[REGEN] Starting regeneration for {len(search_results)} cached search results")

                # Step 1: Re-process search results to get fresh summaries
                summaries = await self.content_processor.process_search_results(
                    query, search_results, RetrievalMode.GOOGLE_SERPER, "gpt-4o-mini", "gpt4o", False  # Use defaults for regeneration
                )

                # Step 2: Generate final summary
                final_summary, result_metadata = await self._generate_final_summary(
                    query, search_results, summaries, "gpt-4o-mini", "gpt4o"  # Use defaults for regeneration
                )

                if final_summary.startswith("Error:"):
                    return final_summary, 0.0, {}

                # Step 3: Update cache with regenerated results
                result_with_summary = {
                    "search_results": search_results,
                    "summaries": final_summary,
                    "summary_metadata": result_metadata
                }

                # Update cache with regenerated summary using strict cache key
                if original_query:
                    # Use same hardcoded params as regeneration (lines 309-310, 314-315)
                    cache_key = self._make_cache_key(
                        original_query,
                        len(search_results),  # top_k = number of results
                        "gpt-4o-mini",        # hardcoded model for regeneration
                        "gpt4o",              # hardcoded prompt for regeneration
                        False,                # hardcoded use_jina=False
                        False                 # hardcoded enable_thinking=False
                    )
                    await self.cache.set(cache_key, result_with_summary)
                    logger.info(f"[REGEN] Updated cache with regenerated summary")

                return f"Regenerated summary for query: {query}\n{final_summary}", 0.0, {"cached": False, "regenerated": True}

            except Exception as e:
                logger.error(f"[REGEN] ✗ Regeneration error: {e}")
                return f"Error: An error occurred during regeneration: {str(e)}", 0.0, {}