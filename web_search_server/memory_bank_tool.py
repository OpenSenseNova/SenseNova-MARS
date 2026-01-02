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
import sqlite3
import json
import hashlib
from datetime import datetime
from typing import Optional, Tuple, Dict
import aiohttp

# Import only what we need - no VERL dependencies
from llm_generator import LLMGenerator

# Import refactored components
from config import SearchConfig, BrowserPoolConfig, RateLimitConfig, HTTPConfig, PageVisitConfig, JinaConfig
from browser_pool import SimpleBrowserPool
from search_providers import GoogleSerperProvider, LocalRetrievalProvider
from content_processor import ContentProcessor
from search_orchestrator import SearchOrchestrator
from progress_tracker import ProgressTracker
from cache import SQLiteCache

logger = logging.getLogger(__name__)
# Default to WARN level for production, override with env var
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


# Helper functions for consistent logging (matching original)
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



class WebSearchAndSummarizeTool:
    """Main tool class that coordinates all search and summarization functionality"""

    def __init__(self, config: dict, top_k: int = 5):
        # Store config directly - no parent class needed
        self.config = config

        # Initialize configurations
        # Parse nested config structure
        search_config_data = config.get('search', {})
        page_visit_config_data = config.get('page_visit', {})
        llm_config_data = config.get('llm', {})

        jina_config_data = page_visit_config_data.get('jina', {})
        jina_config = JinaConfig(
            timeout=jina_config_data.get('timeout', 30),
            max_retries=jina_config_data.get('max_retries', 3)
        )
        page_visit_config = PageVisitConfig(
            jina=jina_config
        )

        self.search_config = SearchConfig(
            top_k=search_config_data.get('top_k', top_k),
            cache_dir=search_config_data.get('cache_dir', './search_cache'),
            llm_config_path=llm_config_data.get('config_path', './llm_config.json'),
            llm_content_limit=llm_config_data.get('content_limit', 100000),
            page_visit=page_visit_config
        )
        
        self.browser_config = BrowserPoolConfig(
            pool_size=64,
            max_retries=3,
            browser_timeout=15.0
        )
        
        # Extract rate limits from config if available
        rate_limits = config.get('rate_limits', {})
        self.rate_config = RateLimitConfig(
            serper_rate=rate_limits.get('serper_rate', 100),
            serper_capacity=rate_limits.get('serper_capacity', 10),
            search_semaphore_limit=rate_limits.get('search_semaphore_limit', 100),
            llm_semaphore_limit=rate_limits.get('llm_semaphore_limit', 1024),
            page_semaphore_limit=rate_limits.get('page_semaphore_limit', 512)
        )
        self.http_config = HTTPConfig()

        # Ensure cache directory exists
        os.makedirs(self.search_config.cache_dir, exist_ok=True)

        # Initialize core components
        self._initialize_components()

        # Add async locks for thread-safe initialization
        self._orchestrator_init_lock = None
        self._http_session_init_lock = None

    def _initialize_components(self):
        """Initialize all components with proper dependency injection"""
        # SQLite cache
        self.sqlite_cache = SQLiteCache(self.search_config.cache_dir)

        # Progress tracker (replaces global counters)
        self.progress_tracker = ProgressTracker()

        # LLM generator
        self.llm_generator = LLMGenerator(config_path=self.search_config.llm_config_path, progress_tracker=self.progress_tracker)

        # Browser pool management
        self.browser_pool = SimpleBrowserPool(self.browser_config)
        
        # Rate limiting semaphores
        self.search_semaphore = asyncio.Semaphore(self.rate_config.search_semaphore_limit)
        self.llm_semaphore = asyncio.Semaphore(self.rate_config.llm_semaphore_limit)
        self.page_semaphore = asyncio.Semaphore(self.rate_config.page_semaphore_limit)
        
        # HTTP session management for concurrent requests
        self.http_session = None
        
        # Initialize orchestrator (will be set up in _ensure_http_session)
        self.orchestrator = None

        # Legacy properties for backwards compatibility
        self.cache_dir = self.search_config.cache_dir
        self.top_k = self.search_config.top_k
        self.google_api_key = os.environ.get('WEBSEARCH_GOOGLE_SERPER_KEY')
        self.search_url = "https://google.serper.dev/search"
        self.excluded_extensions = self.search_config.excluded_extensions

    async def _initialize_orchestrator(self):
        """Initialize the search orchestrator with all dependencies"""
        await self._ensure_http_session()
        
        # Content processor
        content_processor = ContentProcessor(
            browser_pool=self.browser_pool,
            llm_generator=self.llm_generator,
            config=self.search_config,
            progress_tracker=self.progress_tracker,
            page_semaphore=self.page_semaphore,
            llm_semaphore=self.llm_semaphore
        )
        
        # Search providers
        providers = {
            'google_serper': GoogleSerperProvider(
                config=self.search_config,
                rate_config=self.rate_config,
                http_session=self.http_session,
                progress_tracker=self.progress_tracker
            ),
            'local': LocalRetrievalProvider(
                config=self.search_config,
                http_session=self.http_session,
                progress_tracker=self.progress_tracker
            )
        }
        
        # Search orchestrator (now handles cache keys internally with strict format)
        self.orchestrator = SearchOrchestrator(
            providers=providers,
            content_processor=content_processor,
            cache=self.sqlite_cache,
            llm_generator=self.llm_generator,
            config=self.search_config,
            search_semaphore=self.search_semaphore,
            llm_semaphore=self.llm_semaphore
        )

    async def _init_http_session(self):
        """Initialize shared HTTP session with connection pooling for reasonable concurrency"""
        if self.http_session is None:
            # Configure connector to match semaphore limits for optimal concurrency
            connector = aiohttp.TCPConnector(
                limit=self.http_config.connection_pool_size,
                limit_per_host=self.http_config.connection_pool_per_host,
                ttl_dns_cache=self.http_config.dns_cache_ttl,
                use_dns_cache=True,
                keepalive_timeout=self.http_config.keepalive_timeout,
                enable_cleanup_closed=True
            )
            
            # Create session with timeout configuration
            timeout = aiohttp.ClientTimeout(
                total=self.http_config.total_timeout, 
                connect=self.http_config.connect_timeout
            )
            self.http_session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={'User-Agent': 'Mozilla/5.0 (compatible; WebSearchBot/1.0)'}
            )

    async def _ensure_http_session(self):
        """Ensure HTTP session is ready - thread-safe initialization"""
        if self.http_session is None or self.http_session.closed:
            # Create lock on first access
            if self._http_session_init_lock is None:
                self._http_session_init_lock = asyncio.Lock()

            async with self._http_session_init_lock:
                # Double-check pattern to avoid race condition
                if self.http_session is None or self.http_session.closed:
                    await self._init_http_session()

    async def _cleanup_http_session(self):
        """Clean up HTTP session resources"""
        if self.http_session and not self.http_session.closed:
            await self.http_session.close()
        self.http_session = None

    async def execute(self, instance_id: str, parameters: str, **kwargs) -> Tuple[str, float, dict]:
        """Main execution method - delegates to orchestrator"""
        # Thread-safe lazy initialization of orchestrator
        if self.orchestrator is None:
            # Create lock on first access
            if self._orchestrator_init_lock is None:
                self._orchestrator_init_lock = asyncio.Lock()

            async with self._orchestrator_init_lock:
                # Double-check pattern to avoid race condition
                if self.orchestrator is None:
                    await self._initialize_orchestrator()

        # Delegate to orchestrator
        return await self.orchestrator.process_query(parameters)

    # Legacy methods for backwards compatibility
    def _normalize_query(self, query: str) -> str:
        """Normalize query exactly like original for cache compatibility"""
        import re
        
        # Convert to lowercase
        normalized = query.lower()
        
        # Remove extra whitespace (including tabs, newlines)
        normalized = ' '.join(normalized.split())
        
        # Handle common contractions for better matching
        contractions = {
            "what's": "what is",
            "that's": "that is", 
            "there's": "there is",
            "it's": "it is",
            "i'm": "i am",
            "you're": "you are",
            "they're": "they are",
            "we're": "we are",
            "don't": "do not",
            "doesn't": "does not",
            "won't": "will not",
            "can't": "cannot",
        }
        
        for contraction, expansion in contractions.items():
            normalized = normalized.replace(contraction, expansion)
        
        # Normalize punctuation - remove most punctuation but keep meaningful ones
        # Remove trailing punctuation that doesn't change meaning
        normalized = re.sub(r'[.!]+$', '', normalized)  # Remove trailing . ! 
        normalized = re.sub(r'\s*\?\s*$', '', normalized)  # Remove trailing ?
        
        # Normalize multiple spaces to single space
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        return normalized

    def _get_cache_key(self, query: str) -> str:
        """Generate cache key exactly like original - no delegation needed"""
        import re
        
        # Exact same normalization logic as original
        normalized_query = self._normalize_query(query)
        
        # Debug query normalization  
        if query != normalized_query:
            logger.debug(f"[CACHE] Query normalized: '{_truncate_query(query)}' -> '{_truncate_query(normalized_query)}'")
        
        # Include all parameters that affect the result - use default since no config model
        # Cache key will now include dynamic parameters passed in requests
        cache_data = f"{normalized_query}|{self.top_k}"
        
        return hashlib.md5(cache_data.encode('utf-8')).hexdigest()

    async def regenerate_summary_from_cache(self, cached_data: dict, original_query: str = None) -> tuple:
        """
        Regenerate failed summary using existing cached search results.
        REFACTORED: delegates to orchestrator like execute() method.
        """
        # Thread-safe lazy initialization of orchestrator
        if self.orchestrator is None:
            # Create lock on first access
            if self._orchestrator_init_lock is None:
                self._orchestrator_init_lock = asyncio.Lock()

            async with self._orchestrator_init_lock:
                # Double-check pattern to avoid race condition
                if self.orchestrator is None:
                    await self._initialize_orchestrator()

        # Delegate to orchestrator for consistent refactored architecture
        return await self.orchestrator.regenerate_summary_from_cache(cached_data, original_query)

    async def close(self):
        """Explicit cleanup method - call this when done with the tool"""
        await self.browser_pool.close_all()
        await self._cleanup_http_session()
        # Clean up LLM generator to prevent httpx connection leaks
        if hasattr(self.llm_generator, 'close'):
            await self.llm_generator.close()

        # CRITICAL: Close SQLite cache to checkpoint WAL and prevent corruption
        if hasattr(self, 'sqlite_cache') and hasattr(self.sqlite_cache, 'close'):
            self.sqlite_cache.close()

        logger.info("[TOOL] ✓ WebSearchAndSummarizeTool cleanup completed")