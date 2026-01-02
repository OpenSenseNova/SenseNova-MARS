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

from dataclasses import dataclass
from enum import Enum
from typing import Tuple


class RetrievalMode(Enum):
    GOOGLE_SERPER = "google_serper"
    LOCAL = "local"


@dataclass
class BrowserPoolConfig:
    pool_size: int = 64
    max_retries: int = 3
    browser_timeout: float = 15.0
    page_timeout: float = 20.0
    content_timeout: float = 15.0
    pool_health_check_interval: float = 60.0
    emergency_reset_timeout: float = 60.0
    circuit_breaker_threshold: float = 0.9


@dataclass
class JinaConfig:
    timeout: int = 30
    max_retries: int = 3


@dataclass
class PageVisitConfig:
    jina: JinaConfig = None

    def __post_init__(self):
        if self.jina is None:
            self.jina = JinaConfig()


@dataclass
class SearchConfig:
    top_k: int = 5
    cache_dir: str = './search_cache'
    excluded_extensions: Tuple[str, ...] = ('.pdf', '.doc', '.docx', '.ppt', '.pptx', '.xls', '.xlsx', '.jpg', '.jpeg', '.png', '.gif')
    llm_config_path: str = './llm_config.json'
    llm_content_limit: int = 100000
    page_visit: PageVisitConfig = None

    def __post_init__(self):
        if self.page_visit is None:
            self.page_visit = PageVisitConfig()


@dataclass
class RateLimitConfig:
    serper_rate: int = 100
    serper_capacity: int = 10
    search_semaphore_limit: int = 100
    llm_semaphore_limit: int = 1024
    page_semaphore_limit: int = 512


@dataclass
class HTTPConfig:
    total_timeout: float = 20.0
    connect_timeout: float = 5.0
    connection_pool_size: int = 200
    connection_pool_per_host: int = 100
    dns_cache_ttl: int = 300
    keepalive_timeout: int = 30