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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from browser_pool import SimpleBrowserPool

from config import BrowserPoolConfig

logger = logging.getLogger(__name__)


class BrowserHealthMonitor:
    """Monitors browser pool health and triggers recovery when needed"""
    
    def __init__(self, pool: 'SimpleBrowserPool', config: BrowserPoolConfig):
        self.pool = pool
        self.config = config
        self.health_monitor_task = None
        self.health_monitor_active = False
        
    async def start_monitoring(self):
        """Start the proactive health monitoring task"""
        if not self.health_monitor_active and (self.health_monitor_task is None or self.health_monitor_task.done()):
            self.health_monitor_active = True
            self.health_monitor_task = asyncio.create_task(self._health_check_task())
            logger.debug("[BROWSER] ✓ Health monitoring started")

    async def stop_monitoring(self):
        """Stop health monitoring gracefully"""
        self.health_monitor_active = False
        if self.health_monitor_task and not self.health_monitor_task.done():
            try:
                await asyncio.wait_for(self.health_monitor_task, timeout=15.0)
            except asyncio.TimeoutError:
                logger.warning("[BROWSER] Health monitor task did not stop gracefully")
                self.health_monitor_task.cancel()
            except Exception as e:
                logger.debug(f"[BROWSER] Health monitor cleanup error: {e}")

    async def _health_check_task(self):
        """Optimized health monitoring with adaptive intervals and minimal resource usage"""
        last_playwright_test = 0
        consecutive_failures = 0
        last_health_log = 0
        
        try:
            while self.health_monitor_active:
                try:
                    # OPTIMIZED: Adaptive sleep interval based on pool health
                    current_time = time.time()
                    
                    # Quick health check (minimal locking)
                    async with self.pool.pool_lock:
                        healthy_count = self.pool.pool_size - len(self.pool.failed_browsers)
                        health_percentage = (healthy_count / self.pool.pool_size) * 100
                    
                    # Periodic browser health logging (every 5 minutes)
                    if current_time - last_health_log >= 300:  # 5 minutes = 300 seconds
                        logger.info(f"[BROWSER] 🏥 Browser Pool Health - Healthy: {healthy_count}/{self.pool.pool_size} ({health_percentage:.1f}%)")
                        last_health_log = current_time
                    
                    # OPTIMIZED: Adaptive monitoring frequency
                    if health_percentage >= 75:
                        sleep_interval = 60  # Healthy - check less frequently
                    elif health_percentage >= 50:
                        sleep_interval = 45  # Warning - moderate frequency
                    elif health_percentage >= 25:
                        sleep_interval = 30  # Poor - more frequent
                    else:
                        sleep_interval = 20  # Critical - most frequent
                    
                    if health_percentage < 25:  # Critical threshold
                        logger.warning(f"[BROWSER] 🏥 Pool health critical: {health_percentage:.1f}%")
                        
                        # OPTIMIZED: Test Playwright only occasionally, not every check
                        if current_time - last_playwright_test > 120:  # Max every 2 minutes
                            try:
                                if self.pool.playwright:
                                    # OPTIMIZED: Lighter health check - just verify instance is alive
                                    test_browser = await asyncio.wait_for(
                                        self.pool.playwright.chromium.launch(headless=True, args=['--no-sandbox', '--disable-extensions']),
                                        timeout=8.0  # Shorter timeout
                                    )
                                    await test_browser.close()
                                    logger.debug("[BROWSER] ✓ Playwright health check passed")
                                    consecutive_failures = 0
                                    last_playwright_test = current_time
                                else:
                                    logger.warning("[BROWSER] Playwright instance is None - triggering initialization")
                                    await self.pool.initialize()
                                    consecutive_failures = 0
                                    
                            except Exception as health_error:
                                consecutive_failures += 1
                                logger.error(f"[BROWSER] 🏥 Playwright health check failed (#{consecutive_failures}): {health_error}")
                                
                                # OPTIMIZED: Only trigger emergency reset after multiple failures
                                if consecutive_failures >= 3:
                                    logger.critical("[BROWSER] Multiple health check failures - triggering emergency reset")
                                    asyncio.create_task(self.pool.recovery_manager.emergency_playwright_reset())
                                    consecutive_failures = 0  # Reset counter
                                    last_playwright_test = current_time
                    
                    elif health_percentage < 50:  # Warning threshold
                        logger.info(f"[BROWSER] 🏥 Pool health warning: {health_percentage:.1f}% ({healthy_count}/{self.pool.pool_size} healthy)")
                        consecutive_failures = 0  # Reset on recovery
                    else:
                        # Pool is healthy
                        consecutive_failures = 0
                    
                    # OPTIMIZED: Adaptive sleep based on health (with cancellation check)
                    # Break sleep into 1-second chunks to allow responsive shutdown
                    remaining_sleep = sleep_interval
                    while remaining_sleep > 0 and self.health_monitor_active:
                        chunk = min(1.0, remaining_sleep)
                        await asyncio.sleep(chunk)
                        remaining_sleep -= chunk
                        
                except Exception as e:
                    logger.error(f"[BROWSER] Health check iteration error: {e}")
                    # Brief pause before retrying (with cancellation check)
                    for _ in range(10):
                        if not self.health_monitor_active:
                            break
                        await asyncio.sleep(1.0)
                    
        except Exception as e:
            logger.error(f"[BROWSER] Health monitoring task failed: {e}")
        finally:
            self.health_monitor_active = False
            logger.debug("[BROWSER] Health monitoring stopped")