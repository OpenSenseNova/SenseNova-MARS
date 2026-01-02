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
import json
import logging
import os
import shutil
import sqlite3
import time
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


class SQLiteCache:
    """SQLite-based cache for search results"""
    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        self.db_path = os.path.join(cache_dir, 'search_cache.db')

        # Cache hit statistics
        self.cache_hits = 0
        self.cache_misses = 0

        # Ensure cache directory exists
        os.makedirs(cache_dir, exist_ok=True)

        # Check database health and auto-repair if corrupted
        if not self._check_and_repair_db():
            logger.error("[CACHE] Failed to initialize healthy cache")
            raise RuntimeError("Cache database is corrupted and repair failed")

        # Initialize database
        self._init_db()
        logger.info(f"[CACHE] ✓ Initialized: {self.db_path}")

    def _check_and_repair_db(self) -> bool:
        """
        Check database health at startup, auto-repair if corrupted

        Returns:
            True if database is healthy (or successfully repaired)
            False if repair failed
        """
        if not os.path.exists(self.db_path):
            logger.info("[CACHE] No existing cache database, will create new")
            return True

        try:
            # Quick integrity check
            conn = sqlite3.connect(self.db_path, timeout=5.0)
            cursor = conn.execute('PRAGMA quick_check')
            result = cursor.fetchone()[0]
            conn.close()

            if result == "ok":
                logger.info("[CACHE] ✓ Database integrity check passed")
                return True

            # Corruption detected - attempt auto-repair
            logger.warning(f"[CACHE] ⚠️  Corruption detected: {result}")
            return self._auto_repair_db()

        except Exception as e:
            logger.error(f"[CACHE] Health check failed: {e}")
            return self._auto_repair_db()

    def _auto_repair_db(self) -> bool:
        """
        Automatic database repair by row-by-row recovery (optimized version)

        Returns:
            True if repair succeeded, False otherwise
        """
        logger.warning("[CACHE] 🔧 Starting automatic database repair...")

        corrupted_path = self.db_path
        backup_path = f"{self.db_path}.corrupted.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        recovered_path = f"{self.db_path}.recovered"

        try:
            # Step 1: Backup corrupted database
            shutil.copy2(corrupted_path, backup_path)
            logger.info(f"[CACHE] Backed up corrupted DB to: {backup_path}")

            # Step 2: Create recovery database
            source_conn = sqlite3.connect(corrupted_path, timeout=5.0)
            # Disable automatic UTF-8 decoding - we'll decode manually to handle errors
            source_conn.text_factory = bytes
            source_conn.row_factory = sqlite3.Row

            dest_conn = sqlite3.connect(recovered_path, timeout=5.0)
            # Optimized settings for faster recovery
            dest_conn.execute('PRAGMA journal_mode=WAL')
            dest_conn.execute('PRAGMA synchronous=OFF')  # Faster during recovery
            dest_conn.execute('PRAGMA cache_size=-128000')  # 128MB cache for speed

            # Create schema
            dest_conn.execute('''
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            dest_conn.execute('CREATE INDEX IF NOT EXISTS idx_created_at ON cache(created_at)')

            # Step 3: Copy recoverable data with optimized batching
            recovered = 0
            failed = 0
            batch = []
            batch_size = 1000

            logger.info("[CACHE] Recovery progress: starting...")
            cursor = source_conn.execute('SELECT key, value, created_at FROM cache')

            # Iterate with for loop - fetchone() no longer throws UTF-8 errors (returns bytes)
            for row in cursor:
                try:
                    # Manually decode bytes to UTF-8 strings - this is where errors are caught
                    key = row['key'].decode('utf-8') if isinstance(row['key'], bytes) else row['key']
                    value = row['value'].decode('utf-8') if isinstance(row['value'], bytes) else row['value']
                    created_at = row['created_at'].decode('utf-8') if isinstance(row['created_at'], bytes) else row['created_at']

                    batch.append((key, value, created_at))

                    if len(batch) >= batch_size:
                        dest_conn.executemany(
                            'INSERT OR REPLACE INTO cache (key, value, created_at) VALUES (?, ?, ?)',
                            batch
                        )
                        recovered += len(batch)
                        batch = []

                        if recovered % 5000 == 0:
                            dest_conn.commit()
                            logger.info(f"[CACHE] Recovery progress: {recovered:,} entries...")
                except (UnicodeDecodeError, Exception):
                    # Skip corrupted rows (UTF-8 decode errors) - cursor has already advanced
                    failed += 1

            # Insert remaining batch
            if batch:
                try:
                    dest_conn.executemany(
                        'INSERT OR REPLACE INTO cache (key, value, created_at) VALUES (?, ?, ?)',
                        batch
                    )
                    recovered += len(batch)
                except Exception:
                    failed += len(batch)

            dest_conn.commit()

            # Restore normal synchronous mode
            dest_conn.execute('PRAGMA synchronous=NORMAL')

            source_conn.close()
            dest_conn.close()

            # Step 4: Replace corrupted with recovered (atomic operation)
            os.replace(recovered_path, corrupted_path)

            recovery_rate = (recovered / (recovered + failed) * 100) if (recovered + failed) > 0 else 0
            logger.info(f"[CACHE] ✓ Auto-repair complete: recovered {recovered:,} entries, lost {failed:,}")
            logger.info(f"[CACHE] Recovery rate: {recovery_rate:.1f}%")

            return True

        except Exception as e:
            logger.error(f"[CACHE] Auto-repair failed: {e}")
            # Clean up partial recovery
            if os.path.exists(recovered_path):
                try:
                    os.remove(recovered_path)
                except:
                    pass
            return False

    def _init_db(self):
        """Initialize SQLite database with optimized settings"""
        conn = sqlite3.connect(self.db_path, timeout=5.0)
        try:
            # Enable WAL mode for better concurrency
            conn.execute('PRAGMA journal_mode=WAL')
            # Faster synchronization (still safe)
            conn.execute('PRAGMA synchronous=NORMAL')
            # Larger page cache for better performance
            conn.execute('PRAGMA cache_size=-64000')  # 64MB cache
            # Set busy timeout to handle concurrent access
            conn.execute('PRAGMA busy_timeout=5000')  # 5 second timeout
            
            # Create cache table with index
            conn.execute('''
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create index on timestamp for cleanup operations
            conn.execute('CREATE INDEX IF NOT EXISTS idx_created_at ON cache(created_at)')
            
        finally:
            conn.close()
    
    async def get(self, cache_key: str) -> Optional[dict]:
        """Get value from cache by key"""
        def _get_sync():
            conn = sqlite3.connect(self.db_path, timeout=5.0)
            try:
                conn.execute('PRAGMA busy_timeout=5000')
                cursor = conn.execute('SELECT value FROM cache WHERE key = ?', (cache_key,))
                row = cursor.fetchone()
                if row:
                    return json.loads(row[0])
                return None
            finally:
                conn.close()
        
        result = await asyncio.to_thread(_get_sync)
        
        # Update statistics
        if result:
            self.cache_hits += 1
            logger.debug(f"[CACHE] ✓ Hit for key {cache_key[:8]}")
        else:
            self.cache_misses += 1
            logger.debug(f"[CACHE] ✗ Miss for key {cache_key[:8]}")
        
        return result
    
    async def set(self, cache_key: str, value: dict):
        """Set value in cache with automatic cleanup"""
        def _set_sync():
            max_retries = 5
            base_delay = 0.05  # 50ms

            for attempt in range(max_retries + 1):
                conn = sqlite3.connect(self.db_path, timeout=5.0)
                try:
                    conn.execute('PRAGMA busy_timeout=5000')
                    # Insert or replace the cache entry
                    conn.execute(
                        'INSERT OR REPLACE INTO cache (key, value, created_at) VALUES (?, ?, CURRENT_TIMESTAMP)',
                        (cache_key, json.dumps(value))
                    )

                    # Note: Cache accumulates indefinitely - no size limit enforcement

                    conn.commit()
                    return  # Success, exit retry loop
                except sqlite3.OperationalError as e:
                    if "locked" in str(e).lower() and attempt < max_retries:
                        delay = base_delay * (2 ** attempt)  # Exponential backoff
                        time.sleep(delay)
                        logger.debug(f"[CACHE] Database locked, retrying in {delay:.3f}s (attempt {attempt + 1}/{max_retries + 1})")
                    else:
                        raise  # Re-raise if not a lock error or max retries reached
                finally:
                    conn.close()
        
        await asyncio.to_thread(_set_sync)
        logger.debug(f"[CACHE] ✓ Stored key {cache_key[:8]}")
    
    
    def get_hit_rate(self) -> float:
        """Get cache hit rate percentage"""
        total = self.cache_hits + self.cache_misses
        return (self.cache_hits / total * 100) if total > 0 else 0.0
    
    def get_stats(self) -> dict:
        """Get cache statistics"""
        conn = sqlite3.connect(self.db_path, timeout=5.0)
        try:
            conn.execute('PRAGMA busy_timeout=5000')
            cursor = conn.execute('SELECT COUNT(*), SUM(LENGTH(value)) FROM cache')
            count, total_size = cursor.fetchone()
            return {
                'entry_count': count or 0,
                'total_size_bytes': total_size or 0,
                'cache_hits': self.cache_hits,
                'cache_misses': self.cache_misses,
                'hit_rate_percent': self.get_hit_rate(),
                'db_path': self.db_path
            }
        finally:
            conn.close()

    def close(self):
        """
        Properly close the cache database by checkpointing WAL.

        This is CRITICAL to prevent database corruption on shutdown.
        WAL mode keeps changes in -wal file until checkpointed.
        Without this, forceful shutdowns corrupt the database.
        """
        try:
            logger.info("[CACHE] Closing cache and checkpointing WAL...")
            conn = sqlite3.connect(self.db_path, timeout=10.0)
            try:
                # Checkpoint WAL: flush all WAL data to main database
                # TRUNCATE mode: flush + truncate WAL file to 0 bytes
                conn.execute('PRAGMA wal_checkpoint(TRUNCATE)')

                # Ensure changes are synced to disk
                conn.commit()

                logger.info("[CACHE] ✓ WAL checkpoint completed")
            finally:
                conn.close()
        except Exception as e:
            logger.error(f"[CACHE] Error during close: {e}")
            # Don't raise - we're shutting down anyway