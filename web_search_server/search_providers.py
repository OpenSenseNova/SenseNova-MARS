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
import logging
import os
import time
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Any
import aiohttp

from config import SearchConfig, RateLimitConfig
from progress_tracker import ProgressTracker

logger = logging.getLogger(__name__)


def _truncate_query(query: str, max_length: int = 60) -> str:
    """Truncate query text with ellipsis if too long"""
    if len(query) <= max_length:
        return query
    return query[:max_length-3] + "..."


def _truncate_body(body: str, max_length: int = 200) -> str:
    """Truncate response body with ellipsis if too long"""
    if len(body) <= max_length:
        return body
    return body[:max_length-3] + "..."


class SerperRateLimiter:
    """Token bucket rate limiter for Serper API - prevents concurrency bottlenecks"""
    def __init__(self, rate: int = 100, capacity: int = 10):
        self.rate = rate  # tokens per second
        self.capacity = capacity  # burst capacity
        self.tokens = float(capacity)
        self.last_update = time.time()
        self._lock = asyncio.Lock()

    async def acquire(self):
        """Acquire a token for making a Serper API request"""
        while True:
            wait_time = 0
            
            # Critical: BRIEF lock for state updates only
            async with self._lock:
                now = time.time()
                elapsed = now - self.last_update
                
                # Refill tokens based on elapsed time
                self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
                self.last_update = now

                if self.tokens >= 1:
                    self.tokens -= 1
                    return  # Success - no waiting needed
                else:
                    # Calculate wait time for next token
                    wait_time = (1 - self.tokens) / self.rate

            # Critical: Sleep OUTSIDE the lock - allows other requests to proceed
            if wait_time > 0:
                await asyncio.sleep(wait_time)
                # Loop back to try acquiring again


class SearchProvider(ABC):
    """Abstract base class for search providers"""
    
    @abstractmethod
    async def search(self, query: str, top_k: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Perform search and return results.
        
        Returns:
            tuple: (all_results, limited_results) where limited_results is top_k items
        """
        pass


class GoogleSerperProvider(SearchProvider):
    """Google Serper search provider"""
    
    def __init__(self, config: SearchConfig, rate_config: RateLimitConfig, 
                 http_session: aiohttp.ClientSession, progress_tracker: ProgressTracker):
        self.config = config
        self.rate_config = rate_config
        self.http_session = http_session
        self.progress_tracker = progress_tracker
        
        # Search API configuration
        self.google_api_key = os.environ.get('WEBSEARCH_GOOGLE_SERPER_KEY')
        self.search_url = "https://google.serper.dev/search"
        
        # Serper API rate limiting (max 100 req/s with token bucket)
        self._serper_rate_limiter = SerperRateLimiter(
            rate=rate_config.serper_rate, 
            capacity=rate_config.serper_capacity
        )

    async def search(self, query: str, top_k: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Execute Google search and return results"""
        if not self.google_api_key:
            raise ValueError("Google API key not configured.")

        payload = {"q": query}
        headers = {'X-API-KEY': self.google_api_key, 'Content-Type': 'application/json'}

        # Rate limiting: token bucket ensures 100 req/s without blocking concurrency
        await self._serper_rate_limiter.acquire()

        # Track search progress
        self.progress_tracker.update_search_progress(active_delta=1)
        progress = self.progress_tracker.get_progress_snapshot()
        total_search_started = progress['search']['active'] + progress['search']['completed'] + progress['search']['failed']
        logger.debug(f"[SEARCH] Starting #{total_search_started}: {_truncate_query(query)}")
        start_time = time.time()

        try:
            # Use shared session for connection pooling and proper resource management
            async with self.http_session.post(self.search_url, headers=headers, json=payload) as response:
                # Capture response body BEFORE raise_for_status for error details
                response_text = None
                if response.status != 200:
                    try:
                        response_text = await response.text()
                    except:
                        pass

                response.raise_for_status()

                search_data = await response.json()
                organic_results = search_data.get('organic', [])

                elapsed = time.time() - start_time
                self.progress_tracker.update_search_progress(active_delta=-1, completed_delta=1)

                # Log completion with timing (INFO if slow, DEBUG if fast)
                if elapsed > 3.0:
                    logger.info(f"[SEARCH] ✓ Slow search #{total_search_started} completed in {elapsed:.1f}s ({len(organic_results)} results)")
                else:
                    logger.debug(f"[SEARCH] ✓ Completed #{total_search_started} in {elapsed:.1f}s ({len(organic_results)} results)")

                # Progress summary every 10 searches or when active searches drop to 0
                progress = self.progress_tracker.get_progress_snapshot()
                if total_search_started % 10 == 0 or progress['search']['active'] == 0:
                    logger.info(f"[SEARCH] 📊 Progress - Completed: {progress['search']['completed']}, Failed: {progress['search']['failed']}, Active: {progress['search']['active']}")

                search_results = [{'title': res.get('title', ''), 'snippet': res.get('snippet', ''), 'link': res.get('link', '')}
                        for res in organic_results[:min(len(organic_results), top_k)]]
                all_results = [{'title': res.get('title', ''), 'snippet': res.get('snippet', ''), 'link': res.get('link', '')}
                        for res in organic_results]
                return all_results, search_results

        except aiohttp.ClientResponseError as e:
            elapsed = time.time() - start_time
            self.progress_tracker.update_search_progress(active_delta=-1, failed_delta=1)
            logger.error(f"[SEARCH] ✗ HTTP {e.status} failure #{total_search_started} in {elapsed:.1f}s: {e.message}")
            logger.error(f"[SEARCH] Failed query: {_truncate_query(query)}")
            if response_text:
                logger.error(f"[SEARCH] Error response: {_truncate_body(response_text)}")
            # Re-raise to propagate error and prevent training data contamination
            raise RuntimeError(f"Google Serper API HTTP error {e.status}: {e.message}")
        except aiohttp.ClientError as e:
            elapsed = time.time() - start_time
            self.progress_tracker.update_search_progress(active_delta=-1, failed_delta=1)
            logger.error(f"[SEARCH] ✗ Client error #{total_search_started} in {elapsed:.1f}s: {str(e)}")
            logger.debug(f"[SEARCH] Failed query: {_truncate_query(query)}")
            # Re-raise to propagate error and prevent training data contamination
            raise RuntimeError(f"Google Serper API network/connection error: {str(e)}")
        except Exception as e:
            elapsed = time.time() - start_time
            self.progress_tracker.update_search_progress(active_delta=-1, failed_delta=1)
            logger.error(f"[SEARCH] ✗ Unexpected error #{total_search_started} in {elapsed:.1f}s: {str(e)}")
            logger.debug(f"[SEARCH] Failed query: {_truncate_query(query)}")
            # Re-raise to propagate error and prevent training data contamination
            raise RuntimeError(f"Google Serper unexpected error: {str(e)}")


class LocalRetrievalProvider(SearchProvider):
    """Local database retrieval provider"""
    
    def __init__(self, config: SearchConfig, http_session: aiohttp.ClientSession, 
                 progress_tracker: ProgressTracker):
        self.config = config
        self.http_session = http_session
        self.progress_tracker = progress_tracker

    async def search(self, query: str, top_k: int, local_database_address: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Perform local retrieval API call and return results in same format as Google search.
        
        Returns:
            tuple: (all_results, limited_results) - same format as Google search for pipeline compatibility
        """
        # Track search progress (treating local retrieval as a search operation)
        self.progress_tracker.update_search_progress(active_delta=1)
        progress = self.progress_tracker.get_progress_snapshot()
        total_search_started = progress['search']['active'] + progress['search']['completed'] + progress['search']['failed']
        logger.debug(f"[LOCAL] Starting #{total_search_started}: {_truncate_query(query)}")
        start_time = time.time()

        try:
            # Prepare payload for local retrieval API
            payload = {
                "queries": [query],
                "topk": top_k,
                "return_scores": True
            }
            
            # Make request to local retrieval API
            async with self.http_session.post(f"http://{local_database_address}/retrieve", json=payload) as response:
                response.raise_for_status()
                response_data = await response.json()
                retrieval_results = response_data['result'][0]  # Get first query results
                
                elapsed = time.time() - start_time
                self.progress_tracker.update_search_progress(active_delta=-1, completed_delta=1)
                
                # Log completion
                if elapsed > 3.0:
                    logger.info(f"[LOCAL] ✓ Slow retrieval #{total_search_started} completed in {elapsed:.1f}s ({len(retrieval_results)} results)")
                else:
                    logger.debug(f"[LOCAL] ✓ Completed #{total_search_started} in {elapsed:.1f}s ({len(retrieval_results)} results)")
                
                # Progress summary
                progress = self.progress_tracker.get_progress_snapshot()
                if total_search_started % 10 == 0 or progress['search']['active'] == 0:
                    logger.info(f"[LOCAL] 📊 Progress - Completed: {progress['search']['completed']}, Failed: {progress['search']['failed']}, Active: {progress['search']['active']}")

                # Convert local retrieval results to format compatible with existing pipeline
                def _convert_to_search_format(retrieval_result):
                    """Convert local retrieval results to format compatible with existing pipeline"""
                    formatted_results = []
                    for idx, doc_item in enumerate(retrieval_result):
                        content = doc_item['document']['contents']
                        # Extract title from first line, or use default
                        lines = content.split('\n') if content else ['']
                        title = lines[0] if lines[0].strip() else f"Document {idx+1}"
                        
                        formatted_results.append({
                            'title': title,
                            'content': content,  # Store full content directly
                            'doc_id': idx + 1
                            # No snippet - local retrieval uses full content, not artificial snippets
                        })
                    return formatted_results

                # Convert results to search format
                all_results = _convert_to_search_format(retrieval_results)
                limited_results = all_results[:min(len(all_results), top_k)]
                
                return all_results, limited_results
                    
        except Exception as e:
            elapsed = time.time() - start_time
            self.progress_tracker.update_search_progress(active_delta=-1, failed_delta=1)
            logger.error(f"[LOCAL] ✗ Retrieval error #{total_search_started} in {elapsed:.1f}s: {str(e)}")
            logger.debug(f"[LOCAL] Failed query: {_truncate_query(query)}")
            # Re-raise to propagate error and prevent training data contamination
            raise RuntimeError(f"Local retrieval API connection/request failed: {str(e)}")