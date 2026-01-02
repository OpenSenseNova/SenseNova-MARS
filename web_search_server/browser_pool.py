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
import time
from typing import Optional, Dict, Any, Set
from playwright.async_api import async_playwright
from playwright_stealth import Stealth

from config import BrowserPoolConfig
from user_agent_generator import UserAgentGenerator
from browser_health_monitor import BrowserHealthMonitor
from browser_recovery_manager import BrowserRecoveryManager

logger = logging.getLogger(__name__)


class SimpleBrowserPool:
    """Manages a pool of browsers with health monitoring and automatic recovery"""
    
    def __init__(self, config: BrowserPoolConfig):
        self.config = config
        self.pool_size = config.pool_size
        
        # Core browser pool data
        self.browsers = [None] * self.pool_size  # Store browser instances
        self.failed_browsers: Set[int] = set()  # Browser IDs that permanently failed
        self.current_index = 0
        self.pool_lock = asyncio.Lock()
        
        # Single shared Playwright instance for efficiency
        self.playwright = None
        self.stealth_context = None
        self.playwright_lock = asyncio.Lock()  # Prevent race conditions
        
        # Circuit breaker optimization (avoid locking on every get_browser call)
        self.circuit_breaker_open = False
        self.last_circuit_check = 0
        
        # User agent management
        self.user_agents = UserAgentGenerator.generate_realistic_user_agents(self.pool_size)
        
        # Component initialization
        self.health_monitor = BrowserHealthMonitor(self, config)
        self.recovery_manager = BrowserRecoveryManager(self, config)

    async def initialize(self):
        """Initialize the single shared Playwright instance for the entire pool"""
        async with self.playwright_lock:  # Prevent race conditions
            if self.playwright is None:
                logger.debug("[BROWSER] Initializing shared Playwright instance")
                self.stealth_context = Stealth().use_async(async_playwright())
                self.playwright = await self.stealth_context.__aenter__()
                logger.info("[BROWSER] ✓ Shared Playwright instance ready")
                
                # Start health monitoring after successful initialization
                await self.health_monitor.start_monitoring()

    async def get_browser(self) -> Optional[Dict[str, Any]]:
        """Highly optimized browser acquisition with efficient circuit breaker"""
        
        # Check circuit breaker but allow re-evaluation under lock
        if self.circuit_breaker_open:
            # Re-evaluate circuit breaker under lock to allow recovery
            async with self.pool_lock:
                current_time = time.time()
                if current_time - self.last_circuit_check > 5.0:  # Re-check every 5 seconds
                    failure_rate = len(self.failed_browsers) / self.pool_size
                    self.circuit_breaker_open = failure_rate >= self.config.circuit_breaker_threshold
                    self.last_circuit_check = current_time

                    if not self.circuit_breaker_open:
                        logger.info(f"[BROWSER] ✓ Circuit breaker CLOSED - recovery detected ({failure_rate*100:.1f}% failed)")

                # If still open after re-evaluation, return None
                if self.circuit_breaker_open:
                    logger.error(f"[BROWSER] ✗ Circuit breaker OPEN - pool critically damaged ({len(self.failed_browsers)}/{self.pool_size} failed)")
                    return None
        
        # OPTIMIZED: Single lock acquisition for all pool operations
        browser_id_to_create = None
        browser_to_return = None
        
        async with self.pool_lock:
            # Periodic failure-rate check every 5s (allows opening/closing)
            current_time = time.time()
            if current_time - self.last_circuit_check > 5.0:
                failure_rate = len(self.failed_browsers) / self.pool_size
                self.circuit_breaker_open = failure_rate >= self.config.circuit_breaker_threshold
                self.last_circuit_check = current_time
                if self.circuit_breaker_open:
                    logger.error(f"[BROWSER] ✗ Circuit breaker OPEN - {failure_rate*100:.1f}% failed")
                    return None

            # Try to find existing working browser (single loop, single lock)
            for _ in range(self.pool_size):
                browser_id = self.current_index % self.pool_size
                self.current_index += 1
                
                # Skip permanently failed browsers
                if browser_id in self.failed_browsers:
                    continue
                    
                # Return existing browser if available
                if self.browsers[browser_id] is not None:
                    browser_to_return = self.browsers[browser_id]
                    break
                
                # Found slot that needs new browser
                if browser_id_to_create is None:
                    browser_id_to_create = browser_id
        
        # Return existing browser immediately (no additional locking needed)
        if browser_to_return:
            return browser_to_return
        
        # Create new browser if needed (outside lock for maximum concurrency)
        if browser_id_to_create is not None:
            try:
                # OPTIMIZED: Faster timeout for quicker failure detection
                new_browser = await asyncio.wait_for(
                    self._create_browser(browser_id_to_create),
                    timeout=self.config.browser_timeout
                )
                
                # OPTIMIZED: Single lock acquisition for final update
                async with self.pool_lock:
                    # Check if slot is still empty and not failed (prevent concurrent creation leaks)
                    if (browser_id_to_create not in self.failed_browsers and
                        self.browsers[browser_id_to_create] is None):
                        # Slot is available - assign our browser
                        self.browsers[browser_id_to_create] = new_browser
                        # Reset circuit breaker on successful creation
                        if self.circuit_breaker_open:
                            self.circuit_breaker_open = False
                            logger.info("[BROWSER] ✓ Circuit breaker RESET - successful browser creation")
                        return new_browser
                    else:
                        # Slot occupied or failed - return existing browser or cleanup ours
                        if self.browsers[browser_id_to_create] is not None:
                            logger.debug(f"[BROWSER] Slot {browser_id_to_create} occupied during creation - using existing")
                            asyncio.create_task(self._force_cleanup_browser_obj(new_browser))
                            return self.browsers[browser_id_to_create]
                        else:
                            # Slot was marked failed during creation - schedule cleanup
                            logger.debug(f"[BROWSER] Slot {browser_id_to_create} failed during creation")
                            asyncio.create_task(self._force_cleanup_browser_obj(new_browser))
                            return None
                        
            except asyncio.TimeoutError:
                logger.error(f"[BROWSER] ✗ Creation timeout for browser {browser_id_to_create} after {self.config.browser_timeout}s")
                # OPTIMIZED: Batch failure handling
                asyncio.create_task(self.recovery_manager.handle_browser_creation_failure(browser_id_to_create, None))
            except Exception as e:
                logger.warning(f"[BROWSER] ✗ Creation failed for browser {browser_id_to_create}: {e}")
                # OPTIMIZED: Background failure handling to avoid blocking
                asyncio.create_task(self.recovery_manager.handle_browser_creation_failure(browser_id_to_create, e))
                    
        # OPTIMIZED: Quick final check without additional locking
        return None

    async def _create_browser(self, browser_id: int) -> Dict[str, Any]:
        """Create a new browser instance"""
        # Ensure shared Playwright instance is ready
        if self.playwright is None:
            await self.initialize()
        
        # Use the shared Playwright instance - much more efficient!
        browser = await self.playwright.chromium.launch(
            headless=True,
            args=['--no-sandbox', '--disable-dev-shm-usage']
        )
        context = await browser.new_context(
            user_agent=self.user_agents[browser_id],
            viewport={'width': 1920, 'height': 1080},
            extra_http_headers={
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate, br',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            }
        )
        return {
            'id': browser_id,  # Track browser ID for failure marking
            'browser': browser, 
            'context': context,
        }

    async def _cleanup_browser_obj(self, browser_obj: Dict[str, Any]):
        """Clean up resources for a browser object (helper method)"""
        if browser_obj:
            try:
                await browser_obj['context'].close()
                await browser_obj['browser'].close()
            except Exception as e:
                logger.debug(f"[BROWSER] Error cleaning up browser {browser_obj.get('id', 'unknown')}: {e}")

    async def _force_cleanup_browser_obj(self, browser_obj: Dict[str, Any]):
        """Force cleanup of browser object with multiple fallback strategies - NEVER fails"""
        if not browser_obj:
            return
            
        browser_id = browser_obj.get('id', 'unknown')
        
        try:
            # First attempt: normal cleanup
            await self._cleanup_browser_obj(browser_obj)
            return
        except Exception as normal_cleanup_error:
            logger.debug(f"[BROWSER] Normal cleanup failed for browser {browser_id}: {normal_cleanup_error}")
        
        # Fallback 1: Force close context and browser separately
        try:
            if browser_obj.get('context'):
                await asyncio.wait_for(browser_obj['context'].close(), timeout=5.0)
        except Exception as context_error:
            logger.error(f"[BROWSER] ✗ Context force-close failed for browser {browser_id}: {context_error}")
        
        try:
            if browser_obj.get('browser'):
                await asyncio.wait_for(browser_obj['browser'].close(), timeout=5.0)
        except Exception as browser_error:
            logger.error(f"[BROWSER] ✗ Browser force-close failed for browser {browser_id}: {browser_error}")
        
        # Fallback 2: Try to access underlying process (if available) to force kill
        try:
            browser = browser_obj.get('browser')
            if browser and hasattr(browser, '_connection') and browser._connection:
                # Close connection to force cleanup
                try:
                    await browser._connection.dispose()
                except Exception as conn_error:
                    logger.debug(f"[BROWSER] Connection disposal failed for browser {browser_id}: {conn_error}")
        except Exception as process_error:
            logger.debug(f"[BROWSER] Process-level cleanup failed for browser {browser_id}: {process_error}")
        
        logger.debug(f"[BROWSER] ✓ Force cleanup completed for browser {browser_id}")

    async def mark_browser_failed(self, browser_id: int):
        """Mark a browser as permanently failed and trigger self-healing if needed"""
        browser_to_cleanup = None
        
        async with self.pool_lock:
            if browser_id not in self.failed_browsers:
                # Get browser object for cleanup outside the lock
                browser_to_cleanup = self.browsers[browser_id]
                self.browsers[browser_id] = None  # Clear slot immediately
                self.failed_browsers.add(browser_id)
                logger.debug(f"[BROWSER] ✗ Marked browser {browser_id} as failed. Total failed: {len(self.failed_browsers)}")
        
        # --- Perform actions outside the lock ---
        
        # Schedule non-blocking cleanup task
        if browser_to_cleanup:
            asyncio.create_task(self._force_cleanup_browser_obj(browser_to_cleanup))
            logger.debug(f"[BROWSER] Scheduled cleanup for failed browser {browser_id}")
        
        # Check if restart is needed and trigger
        await self.recovery_manager.trigger_restart_if_needed(browser_id)

    async def close_all(self):
        """Clean up all browsers and the shared Playwright instance with guaranteed cleanup"""
        # Stop health monitoring first
        await self.health_monitor.stop_monitoring()
        
        # Close all individual browsers with force cleanup
        cleanup_tasks = []
        for browser_obj in self.browsers:
            if browser_obj:
                cleanup_tasks.append(self._force_cleanup_browser_obj(browser_obj))
        
        # Execute all cleanups concurrently
        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
            logger.info(f"[BROWSER] ✓ Closed browser pool with {len(cleanup_tasks)} browsers")
        
        # Close the shared Playwright instance
        if self.stealth_context:
            try:
                await self.stealth_context.__aexit__(None, None, None)
                logger.info("[BROWSER] ✓ Shared Playwright instance closed")
            except Exception as e:
                logger.error(f"[BROWSER] ✗ Error closing shared Playwright instance: {e}")
        
        self.playwright = None
        self.stealth_context = None