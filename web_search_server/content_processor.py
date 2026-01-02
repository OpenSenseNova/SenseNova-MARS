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
import requests
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup

from config import SearchConfig, RetrievalMode
from browser_pool import SimpleBrowserPool
from progress_tracker import ProgressTracker
from llm_generator import LLMGenerator

logger = logging.getLogger(__name__)


class ContentProcessor:
    """Handles URL fetching, content extraction, and LLM summarization"""
    
    def __init__(self, browser_pool: SimpleBrowserPool, llm_generator: LLMGenerator,
                 config: SearchConfig, progress_tracker: ProgressTracker,
                 page_semaphore: asyncio.Semaphore, llm_semaphore: asyncio.Semaphore):
        self.browser_pool = browser_pool
        self.llm_generator = llm_generator
        self.config = config
        self.progress_tracker = progress_tracker
        self.page_semaphore = page_semaphore
        self.llm_semaphore = llm_semaphore
        
        # Simple counters for monitoring
        self._llm_semaphore_acquires = 0
        self._llm_semaphore_releases = 0

    async def process_search_results(self, query: str, search_results: List[Dict[str, Any]],
                                   retrieval_mode: RetrievalMode, llm_model: str, prompt_type: str, use_jina: bool, enable_thinking: bool = False) -> List[Optional[str]]:
        """Process search results and return summaries"""
        logger.info(f"[CONTENT] Processing with: model={llm_model}, prompt={prompt_type}, jina={use_jina}, thinking={enable_thinking}, mode={retrieval_mode.value}")

        if retrieval_mode == RetrievalMode.LOCAL:
            return await self._process_local_documents(query, search_results, llm_model, prompt_type, enable_thinking)
        else:
            return await self._process_web_urls(query, search_results, llm_model, prompt_type, use_jina, enable_thinking)

    async def _process_local_documents(self, query: str, search_results: List[Dict[str, Any]], llm_model: str, prompt_type: str, enable_thinking: bool = False) -> List[Optional[str]]:
        """Process local retrieval documents directly (no URL fetching needed)"""
        logger.info(f"[LOCAL] Starting per-document LLM processing for {len(search_results)} documents")
        summaries = []
        
        for idx, res in enumerate(search_results):
            doc_id = res.get('doc_id', idx+1)
            logger.debug(f"[LOCAL] Processing document {doc_id} (item {idx+1}/{len(search_results)})")
            try:
                local_content = res.get('content')  # Get content directly from result
                if local_content:
                    content_length = len(local_content)
                    logger.info(f"[LOCAL] Document {doc_id}: content length {content_length} chars, starting LLM summarization")
                    
                    # Use LLM to summarize local content directly
                    await self.llm_semaphore.acquire()
                    self._llm_semaphore_acquires += 1
                    try:
                        llm_start_time = time.time()
                        summary = await self.llm_generator.generate_summary(query, local_content, llm_model, prompt_type, self.config.llm_content_limit, enable_thinking)
                        llm_elapsed = time.time() - llm_start_time
                        
                        if summary and not summary.startswith("summary error"):
                            logger.info(f"[LOCAL] ✓ Document {doc_id} LLM success in {llm_elapsed:.1f}s: {len(summary)} chars")
                            # Log summary content every ~20 documents for debugging
                            if idx % 20 == 0:
                                logger.info(f"[LOCAL] Document {doc_id} summary sample: {summary[:300]}{'...' if len(summary) > 300 else ''}")
                            summaries.append(summary)
                        else:
                            logger.error(f"[LOCAL] ✗ Document {doc_id} LLM failed in {llm_elapsed:.1f}s: {summary}")
                            summaries.append(None)
                    finally:
                        self.llm_semaphore.release()
                        self._llm_semaphore_releases += 1
                else:
                    logger.warning(f"[LOCAL] Document {doc_id}: No content found in result")
                    summaries.append(None)
            except Exception as e:
                logger.error(f"[LOCAL] Exception processing document {doc_id}: {type(e).__name__}: {e}")
                summaries.append(None)
        
        # Log summary of per-document processing
        successful_docs = len([s for s in summaries if s is not None])
        logger.info(f"[LOCAL] Per-document LLM processing complete: {successful_docs}/{len(search_results)} successful")
        return summaries

    async def _process_web_urls(self, query: str, search_results: List[Dict[str, Any]], llm_model: str, prompt_type: str, use_jina: bool, enable_thinking: bool = False) -> List[Optional[str]]:
        """Process web URLs by fetching and summarizing content"""
        tasks = [self._fetch_and_summarize_url_with_retry(query, res['link'], llm_model, prompt_type, use_jina, enable_thinking, max_retries=3)
                 for res in search_results]
        # Use return_exceptions=True to prevent one failure from canceling others
        summaries = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and log them
        filtered_summaries = []
        for i, summary in enumerate(summaries):
            if isinstance(summary, Exception):
                logger.error(f"[LLM] ✗ URL summary failed for {search_results[i]['link']}: {summary}")
                filtered_summaries.append(None)
            else:
                filtered_summaries.append(summary)
        
        return filtered_summaries

    async def _fetch_and_summarize_url_with_retry(self, query: str, url: str, llm_model: str, prompt_type: str, use_jina: bool, enable_thinking: bool = False, max_retries: int = 3) -> Optional[str]:
        """Fetch and summarize single URL with retry logic"""
        for attempt in range(max_retries):
            try:
                result = await self._fetch_and_summarize_url(query, url, llm_model, prompt_type, use_jina, enable_thinking)
                return result
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"[WEB] ✗ Failed to fetch {url} after {max_retries} attempts: {e}")
                    return None
                # Exponential backoff between retries
                backoff_time = min(2 ** attempt, 5)  # Max 5 seconds
                logger.debug(f"[WEB] 🔄 Retry {attempt + 1} for {url} in {backoff_time}s: {e}")
                await asyncio.sleep(backoff_time)
        return None

    async def _jina_fetch_content(self, url: str) -> Optional[str]:
        """Fetch webpage content using Jina Reader API"""
        jina_api_key = os.environ.get('JINA_API_KEY')
        if not self.config.page_visit.use_jina or not jina_api_key:
            return None

        jina_config = self.config.page_visit.jina
        headers = {
            "Authorization": f"Bearer {jina_api_key}",
        }

        for attempt in range(jina_config.max_retries):
            try:
                # Use asyncio.to_thread to run requests in a thread pool
                response = await asyncio.to_thread(
                    requests.get,
                    f"https://r.jina.ai/{url}",
                    headers=headers,
                    timeout=jina_config.timeout
                )
                if response.status_code == 200:
                    content = response.text
                    logger.debug(f"[JINA] ✓ Successfully fetched content from {url}")
                    return content
                else:
                    logger.warning(f"[JINA] ✗ HTTP {response.status_code} for {url}: {response.text}")
                    if response.status_code >= 400:
                        break  # Don't retry on client errors

            except Exception as e:
                if attempt == jina_config.max_retries - 1:
                    logger.error(f"[JINA] ✗ Failed to fetch {url} after {jina_config.max_retries} attempts: {e}")
                    return None
                else:
                    backoff_time = min(2 ** attempt, 5)
                    logger.debug(f"[JINA] 🔄 Retry {attempt + 1} for {url} in {backoff_time}s: {e}")
                    await asyncio.sleep(backoff_time)

        return None

    async def _fetch_and_summarize_url(self, query: str, url: str, llm_model: str, prompt_type: str, use_jina: bool, enable_thinking: bool = False) -> Optional[str]:
        """Fetch single URL content and summarize with LLM"""
        if any(url.lower().endswith(ext) for ext in self.config.excluded_extensions):
            logger.debug(f"[WEB] Skipped non-HTML resource: {url}")
            return None

        # Check if Jina is enabled and should be used
        if use_jina and os.environ.get('JINA_API_KEY'):
            logger.info(f"[JINA] Using Jina API for {url}")
            raw_content = await self._jina_fetch_content(url)
            if raw_content:
                # LLM processing for Jina content
                try:
                    await self.llm_semaphore.acquire()
                    self._llm_semaphore_acquires += 1
                    try:
                        summary = await self.llm_generator.generate_summary(query, raw_content, llm_model, prompt_type, self.config.llm_content_limit, enable_thinking)
                        return summary if not summary.startswith("summary error") else None
                    finally:
                        self.llm_semaphore.release()
                        self._llm_semaphore_releases += 1
                except Exception as llm_error:
                    logger.error(f"[LLM] ✗ Summarization error for {url}: {llm_error}")
                    return None
            else:
                logger.warning(f"[JINA] ✗ Failed to fetch content from {url}, skipping browser fallback")
                return None

        # Use browser-based fetching (Jina not enabled or failed)
        logger.info(f"[BROWSER] Using browser for {url} (jina={use_jina}, api_key_present={bool(os.environ.get('JINA_API_KEY'))})")

        # Get browser first (outside semaphore to avoid holding it unnecessarily)
        try:
            browser_obj = await asyncio.wait_for(
                self.browser_pool.get_browser(),
                timeout=10.0
            )
            if browser_obj is None:
                logger.error(f"[BROWSER] ✗ Pool exhausted for URL: {url}")
                return None
        except asyncio.TimeoutError:
            logger.error(f"[BROWSER] ⏰ Acquisition timeout for URL: {url}")
            return None

        # Initialize variables outside semaphore block for LLM processing
        page = None
        raw_content = None
        web_completed = False
        web_failed = False
        web_bot_blocked = False
        start_time = None
        total_started = None

        # Only acquire semaphore after we have a browser
        async with self.page_semaphore:
            
            try:
                page = await browser_obj['context'].new_page()
                
                # Set up resource blocking and tracking patterns
                await self._setup_page_blocking(page)
                
                # Track web request progress - INCREMENT COUNTER
                progress_before = self.progress_tracker.get_progress_snapshot()
                total_before = progress_before['web']['active'] + progress_before['web']['completed'] + progress_before['web']['failed'] + progress_before['web']['bot_blocked']
                
                self.progress_tracker.update_web_progress(active_delta=1)
                total_started = total_before + 1
                logger.debug(f"[WEB] Starting #{total_started}: {url}")
                
                start_time = time.time()
                try:
                    response = await asyncio.wait_for(
                        page.goto(
                            url,
                            timeout=20000,
                            wait_until='domcontentloaded'
                        ),
                        timeout=25.0  # Hard timeout at the asyncio level
                    )
                    elapsed = time.time() - start_time
                    
                    # Check status code before updating progress counters
                    status_code = response.status
                    if status_code >= 400:
                        # Detect bot blocking vs real failures
                        bot_detection_codes = {403, 429, 406, 418, 421, 451}
                        if status_code in bot_detection_codes:
                            web_bot_blocked = True
                            logger.warning(f"[WEB] 🤖 Bot blocked #{total_started} in {elapsed:.1f}s - HTTP {status_code}: {url}")
                        else:
                            web_failed = True
                            logger.warning(f"[WEB] ✗ Failed #{total_started} in {elapsed:.1f}s - HTTP {status_code}: {url}")
                        return None
                    
                    # Web request completed successfully
                    web_completed = True
                    
                    logger.debug(f"[WEB] ✓ Completed #{total_started} in {elapsed:.1f}s{f' (slow)' if elapsed > 5.0 else ''}")
                            
                except Exception as e:
                    elapsed = time.time() - start_time
                    web_failed = True
                    logger.warning(f"[WEB] ✗ Failed #{total_started} in {elapsed:.1f}s: {str(e)} - {url}")
                    return None  # Don't raise - handle all failures here

                # Check content type
                content_type = response.headers.get('content-type', '')
                if 'text/html' not in content_type:
                    logger.debug(f"[WEB] Skipped non-HTML content: {url}")
                    return None
                
                # Get the HTML content after JavaScript execution with timeout
                try:
                    html = await asyncio.wait_for(page.content(), timeout=15.0)
                except asyncio.TimeoutError:
                    elapsed = time.time() - start_time
                    logger.warning(f"[WEB] ⏰ Content timeout #{total_started} after 15s (total: {elapsed:.1f}s): {url}")
                    return None
                
                # Parse with BeautifulSoup
                soup = BeautifulSoup(html, "lxml")
                for script in soup(["script", "style"]):
                    script.decompose()

                raw_content = soup.get_text(separator='\n', strip=True)

            except Exception as e:
                logger.error(f"[WEB] ✗ URL error for {url} [{type(e).__name__}]: {e}")
                
                # Set failure flag - decrement will happen in finally block
                web_failed = True
                
                # If browser_obj exists and this looks like a browser failure, mark it as failed
                if browser_obj and ("browser" in str(e).lower() or "context" in str(e).lower()):
                    browser_id = browser_obj.get('id')
                    if browser_id is not None:
                        await self.browser_pool.mark_browser_failed(browser_id)
                        logger.debug(f"[BROWSER] ✗ Marked browser {browser_id} as failed due to error: {e}")
                    else:
                        # Orphaned browser object - force cleanup immediately
                        logger.debug("[BROWSER] Cleaning up orphaned browser object after failure")
                        await self.browser_pool._force_cleanup_browser_obj(browser_obj)
                return None
            finally:
                # BULLETPROOF: Single decrement point - exactly one decrement per increment
                if total_started is not None:  # Only decrement if we incremented
                    if web_completed:
                        self.progress_tracker.update_web_progress(active_delta=-1, completed_delta=1)
                        logger.debug(f"[WEB] ✓ Decremented active, incremented completed for: {url}")
                    elif web_bot_blocked:
                        self.progress_tracker.update_web_progress(active_delta=-1, bot_blocked_delta=1)
                        logger.debug(f"[WEB] ✓ Decremented active, incremented bot_blocked for: {url}")
                    elif web_failed:
                        self.progress_tracker.update_web_progress(active_delta=-1, failed_delta=1)
                        logger.debug(f"[WEB] ✓ Decremented active, incremented failed for: {url}")
                    else:
                        # Fallback: if no flag set but we incremented, count as failed
                        self.progress_tracker.update_web_progress(active_delta=-1, failed_delta=1)
                        logger.warning(f"[WEB] ⚠️ Fallback decrement (no flag set) for: {url}")
                        
                    # Log final progress if it's a milestone or active drops to 0
                    if start_time is not None:
                        elapsed_total = time.time() - start_time
                        progress = self.progress_tracker.get_progress_snapshot()
                        if total_started % 20 == 0 or progress['web']['active'] == 0:
                            try:
                                async with self.browser_pool.pool_lock:
                                    healthy_browsers = self.browser_pool.pool_size - len(self.browser_pool.failed_browsers)
                                logger.info(f"[WEB] 📊 Progress - Completed: {progress['web']['completed']}, Failed: {progress['web']['failed']}, Bot Blocked: {progress['web']['bot_blocked']}, Active: {progress['web']['active']} | Browsers: {healthy_browsers}/{self.browser_pool.pool_size} healthy")
                            except:
                                logger.info(f"[WEB] 📊 Progress - Completed: {progress['web']['completed']}, Failed: {progress['web']['failed']}, Bot Blocked: {progress['web']['bot_blocked']}, Active: {progress['web']['active']}")
                                
                # Always close the page to free memory
                if page:
                    try:
                        await page.close()
                    except:
                        pass

        # LLM processing happens outside the page semaphore to avoid blocking page slots
        if web_completed and raw_content is not None:
            try:
                # Acquire semaphore for URL processing
                await self.llm_semaphore.acquire()
                self._llm_semaphore_acquires += 1

                try:
                    summary = await self.llm_generator.generate_summary(query, raw_content, llm_model, prompt_type, self.config.llm_content_limit, enable_thinking)
                    return summary if not summary.startswith("summary error") else None
                finally:
                    self.llm_semaphore.release()
                    self._llm_semaphore_releases += 1

            except Exception as llm_error:
                logger.error(f"[LLM] ✗ Summarization error for {url}: {llm_error}")
                return None

        # If we didn't get content successfully, return None
        return None

    async def _setup_page_blocking(self, page):
        """Set up resource blocking and tracking domain blocking for a page"""
        # Block unnecessary resources while keeping JS for dynamic content
        excluded_resource_types = ["image", "stylesheet", "font", "media", "websocket", "eventsource", "manifest"]
        async def resource_handler(route):
            if route.request.resource_type in excluded_resource_types:
                await route.abort()
            else:
                await route.continue_()

        await page.route("**/*", resource_handler)
        
        # Block tracking and ad domains that perform advanced bot detection
        tracking_domains = [
            # Google tracking and ads
            "*google-analytics.com*", "*googletagmanager.com*", "*doubleclick.net*",
            "*googleadservices.com*", "*googlesyndication.com*", "*googletagservices.com*",
            
            # Facebook/Meta tracking
            "*facebook.com/tr*", "*connect.facebook.net*", "*facebook.net*",
            
            # Other major tracking networks
            "*hotjar.com*", "*mixpanel.com*", "*segment.com*", "*amplitude.com*",
            "*fullstory.com*", "*logrocket.com*", "*mouseflow.com*",
            
            # Ad networks and exchanges
            "*adsystem.com*", "*pubmatic.com*", "*rubiconproject.com*",
            "*amazon-adsystem.com*", "*adsafeprotected.com*",
            
            # Analytics and tracking
            "*newrelic.com*", "*nr-data.net*", "*pingdom.net*",
            "*optimizely.com*", "*quantserve.com*", "*scorecardresearch.com*"
        ]
        
        # Block each tracking domain pattern
        async def tracking_handler(route):
            await route.abort()
        
        for domain_pattern in tracking_domains:
            try:
                await page.route(domain_pattern, tracking_handler)
            except Exception as e:
                logger.debug(f"[WEB] Failed to block tracking domain {domain_pattern}: {e}")
                pass  # Continue if route registration fails

    def format_snippet_content(self, query: str, search_results: List[Dict[str, Any]], 
                              retrieval_mode: RetrievalMode) -> str:
        """Format content as fallback strategy - supports Google Serper and local retrieval"""
        content = f"Search Query: {query}\n\n"
        
        # Use explicit retrieval mode instead of fragile data structure detection
        if retrieval_mode == RetrievalMode.LOCAL:
            # Local retrieval fallback - use full content for LLM processing (no truncation)
            content += "--- Local Documents (Full Content) ---\n"
            for idx, result in enumerate(search_results):
                content += f"[{idx+1}] Title: {result['title']}\n"
                content += f"    Document ID: {result.get('doc_id', idx+1)}\n"
                content += f"    Content: {result['content']}\n\n"
        else:
            # Google Serper fallback - use snippets  
            content += "--- Search Results (Snippets Only) ---\n"
            for idx, result in enumerate(search_results):
                content += f"[{idx+1}] Title: {result['title']}\n"
                content += f"    Snippet: {result.get('snippet', 'No snippet')}\n"
                content += f"    Link: {result.get('link', 'No link')}\n\n"
        return content

    def format_all_content(self, query: str, search_results: List[Dict[str, Any]],
                          summaries: List[Optional[str]]) -> str:
        """Format all content including search results and summaries"""
        all_content = f"Search Query: {query}\n\n"

        # Detect retrieval mode from search results structure
        is_local = search_results and 'doc_id' in search_results[0] if search_results else False
        if is_local:
            # Local retrieval format
            all_content += "--- Local Documents ---\n"
            for idx, result in enumerate(search_results):
                all_content += f"[{idx+1}] Title: {result['title']}\n"
                all_content += f"    Document ID: {result.get('doc_id', idx+1)}\n\n"

            all_content += "--- Document Summaries ---\n"
            for summary in summaries:
                if summary:
                    all_content += f"{summary}\n\n"
        else:
            # Google Serper format
            all_content += "--- Search Results ---\n"
            for idx, result in enumerate(search_results):
                all_content += f"[{idx+1}] Title: {result['title']}\n"
                all_content += f"    Snippet: {result.get('snippet', 'No snippet')}\n"
                all_content += f"    Link: {result.get('link', 'No link')}\n\n"

            all_content += "--- Web Page Summaries ---\n"
            for summary in summaries:
                if summary:
                    all_content += f"{summary}\n\n"

        return all_content
