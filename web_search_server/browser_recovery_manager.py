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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from browser_pool import SimpleBrowserPool

from config import BrowserPoolConfig

logger = logging.getLogger(__name__)


class BrowserRecoveryManager:
    """Handles browser failure recovery and emergency resets"""
    
    def __init__(self, pool: 'SimpleBrowserPool', config: BrowserPoolConfig):
        self.pool = pool
        self.config = config
        
        # Prevent duplicate restart tasks
        self.restart_task = None
        self.restart_lock = asyncio.Lock()
        
        # Emergency reset coordination (prevents concurrent resets)
        self.emergency_reset_lock = asyncio.Lock()
        self.emergency_reset_in_progress = False

    async def handle_browser_creation_failure(self, browser_id: int, error: Exception = None):
        """Handle browser creation failures efficiently in background"""
        try:
            # Mark browser as failed
            async with self.pool.pool_lock:
                self.pool.failed_browsers.add(browser_id)
            
            # Handle Playwright connection failures
            if error:
                error_str = str(error).lower()
                playwright_failure_indicators = [
                    "handler is closed", "target closed", "connection closed", 
                    "browser has been closed", "playwright connection"
                ]
                
                if any(indicator in error_str for indicator in playwright_failure_indicators):
                    logger.critical(f"[BROWSER] Playwright failure detected: {error}")
                    # Trigger emergency reset in background (non-blocking)
                    asyncio.create_task(self.emergency_playwright_reset())
                    
        except Exception as e:
            logger.error(f"[BROWSER] Error in failure handler: {e}")

    async def restart_browsers(self):
        """Enhanced background browser restart with Playwright instance recovery"""
        try:
            # Get list of browsers to restart (quick lock)
            failed_to_restart = []
            async with self.pool.pool_lock:
                healthy_count = self.pool.pool_size - len(self.pool.failed_browsers)
                if healthy_count >= self.pool.pool_size // 2:
                    return  # Another task already fixed it
                    
                failed_to_restart = list(self.pool.failed_browsers)[:self.pool.pool_size // 4]
                logger.info(f"[BROWSER] 🔄 Restarting {len(failed_to_restart)} failed browsers ({healthy_count}/{self.pool.pool_size} healthy)")
            
            restarted_count = 0
            consecutive_playwright_failures = 0  # Track cascade failures
            
            # Restart browsers one by one (no lock held during creation)
            for browser_id in failed_to_restart:
                try:
                    # Clean up old browser (quick lock) - get browser object for force cleanup
                    old_browser_obj = None
                    async with self.pool.pool_lock:
                        if self.pool.browsers[browser_id]:
                            old_browser_obj = self.pool.browsers[browser_id]
                            self.pool.browsers[browser_id] = None
                    
                    # Force cleanup outside the lock
                    if old_browser_obj:
                        await self.pool._force_cleanup_browser_obj(old_browser_obj)
                    
                    # Create new browser (NO LOCK - this is the slow part)
                    logger.debug(f"[BROWSER] 🔄 Restarting browser {browser_id}...")
                    new_browser = await asyncio.wait_for(
                        self.pool._create_browser(browser_id),
                        timeout=30.0
                    )
                    
                    # Update browser pool (quick lock)
                    async with self.pool.pool_lock:
                        self.pool.browsers[browser_id] = new_browser
                        self.pool.failed_browsers.discard(browser_id)
                        restarted_count += 1
                        logger.debug(f"[BROWSER] ✓ Restarted browser {browser_id}")
                    
                    consecutive_playwright_failures = 0  # Reset on success
                        
                except Exception as e:
                    logger.error(f"[BROWSER] ✗ Background restart failed for browser {browser_id}: {e}")
                    
                    # ENHANCED: Detect ALL Playwright connection failures
                    playwright_failure_indicators = [
                        "handler is closed", "target closed", "connection closed",
                        "browser has been closed", "playwright connection",
                        "chrome didn't start", "cdp websocket", "transport endpoint"
                    ]
                    
                    error_str = str(e).lower()
                    is_playwright_failure = any(indicator in error_str for indicator in playwright_failure_indicators)
                    
                    if is_playwright_failure:
                        consecutive_playwright_failures += 1
                        logger.critical(f"[BROWSER] Playwright failure #{consecutive_playwright_failures}: {e}")
                        
                        # ENHANCED: Circuit breaker pattern - progressive recovery
                        if consecutive_playwright_failures >= 3:
                            logger.critical("[BROWSER] Multiple Playwright failures - triggering FULL RESET")
                            
                            try:
                                # Prevent concurrent resets with timeout
                                success = await asyncio.wait_for(self.emergency_playwright_reset(), timeout=60.0)
                                if success:
                                    logger.info("[BROWSER] ✓ Emergency reset successful, retrying browser creation")
                                    consecutive_playwright_failures = 0
                                    continue
                                else:
                                    logger.critical("[BROWSER] ✗ Emergency reset FAILED - pool compromised")
                                    break
                            except asyncio.TimeoutError:
                                logger.critical("[BROWSER] ✗ Emergency reset TIMEOUT - system compromised")
                                break
                        else:
                            # Single failure - shorter backoff before trying next browser
                            await asyncio.sleep(1.0)
                    
                    # Keep in failed set
                    async with self.pool.pool_lock:
                        self.pool.browsers[browser_id] = None
            
            # Final status (quick lock)
            async with self.pool.pool_lock:
                final_healthy = self.pool.pool_size - len(self.pool.failed_browsers)
                logger.info(f"[BROWSER] ✓ Background restart complete: {restarted_count}/{len(failed_to_restart)} restarted. "
                           f"Healthy: {final_healthy}/{self.pool.pool_size}")
                           
        except Exception as e:
            logger.error(f"[BROWSER] ✗ Background browser restart failed: {e}")

    async def emergency_playwright_reset(self) -> bool:
        """Emergency reset of shared Playwright instance - RACE CONDITION SAFE"""
        
        # CRITICAL: Prevent concurrent emergency resets
        async with self.emergency_reset_lock:
            if self.emergency_reset_in_progress:
                logger.info("[BROWSER] Emergency reset already in progress, skipping duplicate")
                return False
            
            self.emergency_reset_in_progress = True
            
        logger.warning("[BROWSER] 🚨 EMERGENCY RESET: Playwright instance recovery starting")
        
        try:
            # Step 1: Force cleanup of ALL browsers concurrently with timeout
            cleanup_tasks = []
            async with self.pool.pool_lock:
                for i, browser_obj in enumerate(self.pool.browsers):
                    if browser_obj:
                        cleanup_tasks.append(self.pool._force_cleanup_browser_obj(browser_obj))
                        self.pool.browsers[i] = None  # Clear immediately
                
                # Mark all as failed temporarily - they'll be recovered after reset
                self.pool.failed_browsers = set(range(self.pool.pool_size))
            
            # Concurrent cleanup with timeout protection
            if cleanup_tasks:
                await asyncio.wait_for(
                    asyncio.gather(*cleanup_tasks, return_exceptions=True),
                    timeout=30.0
                )
                logger.info(f"[BROWSER] ✓ Cleaned up {len(cleanup_tasks)} browsers")
            
            # Step 2: Force close and recreate Playwright instance
            if self.pool.stealth_context:
                try:
                    await asyncio.wait_for(
                        self.pool.stealth_context.__aexit__(None, None, None),
                        timeout=15.0
                    )
                except Exception as close_error:
                    logger.warning(f"[BROWSER] Playwright close error (expected): {close_error}")
            
            # Reset state
            self.pool.playwright = None
            self.pool.stealth_context = None
            
            # Step 3: Initialize fresh instance with retry
            for attempt in range(3):
                try:
                    await asyncio.wait_for(self.pool.initialize(), timeout=30.0)
                    
                    # Verify by creating a test browser
                    test_browser = await asyncio.wait_for(
                        self.pool.playwright.chromium.launch(headless=True, args=['--no-sandbox']),
                        timeout=15.0
                    )
                    await test_browser.close()
                    
                    logger.info("[BROWSER] ✓ EMERGENCY RESET SUCCESSFUL - Playwright instance restored")
                    
                    # Clear failed browsers - they can be recreated on demand
                    async with self.pool.pool_lock:
                        self.pool.failed_browsers.clear()
                    
                    return True
                    
                except Exception as init_error:
                    logger.error(f"[BROWSER] Reset attempt {attempt + 1} failed: {init_error}")
                    if attempt < 2:
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
                        
            logger.critical("[BROWSER] ✗ EMERGENCY RESET FAILED after 3 attempts")
            return False
            
        except asyncio.TimeoutError:
            logger.critical("[BROWSER] ✗ Emergency reset TIMEOUT - system may be compromised")
            return False
        except Exception as e:
            logger.critical(f"[BROWSER] ✗ Emergency reset EXCEPTION: {e}")
            return False
        finally:
            # CRITICAL: Always reset the in-progress flag
            self.emergency_reset_in_progress = False

    async def trigger_restart_if_needed(self, browser_id: int):
        """Trigger restart task if failure threshold is crossed"""
        restart_needed = False
        
        async with self.pool.pool_lock:
            # Check if the pool has crossed the failure threshold
            healthy_count = self.pool.pool_size - len(self.pool.failed_browsers)
            if healthy_count < self.pool.pool_size // 2:
                restart_needed = True
        
        # If the failure threshold was crossed, trigger the self-healing task
        if restart_needed:
            async with self.restart_lock:
                if self.restart_task is None or self.restart_task.done():
                    self.restart_task = asyncio.create_task(self.restart_browsers())
                    logger.warning(f"[BROWSER] 🔄 Pool health critical, triggering background restart")
                else:
                    logger.debug(f"[BROWSER] Background restart already in progress, skipping")