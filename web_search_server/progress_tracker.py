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

import threading
from typing import Dict


class ProgressTracker:
    def __init__(self):
        self._web_progress = {'active': 0, 'completed': 0, 'failed': 0, 'bot_blocked': 0}
        self._search_progress = {'active': 0, 'completed': 0, 'failed': 0}
        self._llm_progress = {'active': 0, 'completed': 0, 'failed': 0, 'retries': 0, 'total_prompt_tokens': 0, 'total_completion_tokens': 0}
        self._lock = threading.Lock()

    def update_web_progress(self, active_delta: int = 0, completed_delta: int = 0, 
                           failed_delta: int = 0, bot_blocked_delta: int = 0):
        """Thread-safe update of web progress counters"""
        with self._lock:
            self._web_progress['active'] += active_delta
            self._web_progress['completed'] += completed_delta
            self._web_progress['failed'] += failed_delta
            self._web_progress['bot_blocked'] += bot_blocked_delta

    def update_search_progress(self, active_delta: int = 0, completed_delta: int = 0,
                              failed_delta: int = 0):
        """Thread-safe update of search progress counters"""
        with self._lock:
            self._search_progress['active'] += active_delta
            self._search_progress['completed'] += completed_delta
            self._search_progress['failed'] += failed_delta

    def update_llm_progress(self, active_delta: int = 0, completed_delta: int = 0,
                           failed_delta: int = 0, retries_delta: int = 0,
                           prompt_tokens_delta: int = 0, completion_tokens_delta: int = 0):
        """Thread-safe update of LLM progress counters"""
        with self._lock:
            self._llm_progress['active'] += active_delta
            self._llm_progress['completed'] += completed_delta
            self._llm_progress['failed'] += failed_delta
            self._llm_progress['retries'] += retries_delta
            self._llm_progress['total_prompt_tokens'] += prompt_tokens_delta
            self._llm_progress['total_completion_tokens'] += completion_tokens_delta

    def get_progress_snapshot(self) -> Dict[str, Dict[str, int]]:
        """Get thread-safe snapshot of progress counters"""
        with self._lock:
            return {
                'web': dict(self._web_progress),
                'search': dict(self._search_progress),
                'llm': dict(self._llm_progress)
            }

    def reset_progress(self):
        """Reset all progress counters"""
        with self._lock:
            self._web_progress = {'active': 0, 'completed': 0, 'failed': 0, 'bot_blocked': 0}
            self._search_progress = {'active': 0, 'completed': 0, 'failed': 0}
            self._llm_progress = {'active': 0, 'completed': 0, 'failed': 0, 'retries': 0, 'total_prompt_tokens': 0, 'total_completion_tokens': 0}