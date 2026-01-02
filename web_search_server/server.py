#!/usr/bin/env python3

import os
import json
import logging
import asyncio
import time
from datetime import datetime
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field
from typing import Optional
import sys
from playwright.async_api import async_playwright

# # Add current and parent directories to path for imports
# current_dir = os.path.dirname(__file__)
# parent_dir = os.path.dirname(current_dir)
# sys.path.insert(0, current_dir)   # Current directory first
# sys.path.insert(0, parent_dir)    # Parent directory second

# Import the tool and dependencies
from memory_bank_tool import WebSearchAndSummarizeTool
from llm_generator import LLMGenerator, cleanup_shared_resources

# Set up logging with file output in logs directory
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = os.path.join(LOG_DIR, f"web_search_server_{timestamp}.log")
logging.basicConfig(
    level=os.getenv("VERL_LOGGING_LEVEL", "INFO"),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()  # Also log to console
    ]
)

# Suppress noisy HTTP logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3.connectionpool").setLevel(logging.WARNING)

# Set memory_bank_tool to INFO for web scraping visibility
logging.getLogger("memory_bank_tool").setLevel(logging.INFO)

logger = logging.getLogger(__name__)

def get_web_progress_stats():
    """Get web scraping progress statistics from the tool's progress tracker"""
    if not shared_memory_bank or not hasattr(shared_memory_bank, 'progress_tracker'):
        # Fallback for when tool is not initialized or doesn't have progress tracker
        return {
            "active_fetches": 0,
            "completed_fetches": 0,
            "failed_fetches": 0,
            "bot_blocked_fetches": 0,
            "total_attempts": 0,
            "success_rate_percent": 0,
            "bot_detection_rate_percent": 0
        }

    progress = shared_memory_bank.progress_tracker.get_progress_snapshot()
    web_progress = progress['web']

    total_attempts = web_progress['completed'] + web_progress['failed'] + web_progress['bot_blocked']
    success_rate = (web_progress['completed'] / max(1, total_attempts)) * 100
    bot_detection_rate = (web_progress['bot_blocked'] / max(1, total_attempts)) * 100

    return {
        "active_fetches": web_progress["active"],
        "completed_fetches": web_progress["completed"],
        "failed_fetches": web_progress["failed"],
        "bot_blocked_fetches": web_progress["bot_blocked"],
        "total_attempts": total_attempts,
        "success_rate_percent": round(success_rate, 2),
        "bot_detection_rate_percent": round(bot_detection_rate, 2)
    }

def get_llm_progress_stats():
    """Get LLM generation progress statistics from the tool's progress tracker"""
    if not shared_memory_bank or not hasattr(shared_memory_bank, 'progress_tracker'):
        # Fallback for when tool is not initialized or doesn't have progress tracker
        return {
            "active_generations": 0,
            "completed_generations": 0,
            "failed_generations": 0,
            "total_retries": 0,
            "total_attempts": 0,
            "success_rate_percent": 0,
            "avg_retries_per_call": 0,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_tokens": 0
        }

    progress = shared_memory_bank.progress_tracker.get_progress_snapshot()
    llm_progress = progress['llm']

    total_attempts = llm_progress['completed'] + llm_progress['failed']
    success_rate = (llm_progress['completed'] / max(1, total_attempts)) * 100
    avg_retries = (llm_progress['retries'] / max(1, total_attempts))
    total_tokens = llm_progress['total_prompt_tokens'] + llm_progress['total_completion_tokens']

    return {
        "active_generations": llm_progress["active"],
        "completed_generations": llm_progress["completed"],
        "failed_generations": llm_progress["failed"],
        "total_retries": llm_progress["retries"],
        "total_attempts": total_attempts,
        "success_rate_percent": round(success_rate, 2),
        "avg_retries_per_call": round(avg_retries, 2),
        "total_prompt_tokens": llm_progress["total_prompt_tokens"],
        "total_completion_tokens": llm_progress["total_completion_tokens"],
        "total_tokens": total_tokens
    }

# Configuration
CONFIG_FILE = os.getenv('WEB_SERVER_CONFIG_FILE', os.path.join(os.path.dirname(__file__), 'config.json'))
CACHE_DIR = os.getenv('WEB_SERVER_CACHE_DIR', os.path.join(os.path.dirname(__file__), 'search_cache'))

# Global tool instance
shared_memory_bank: Optional[WebSearchAndSummarizeTool] = None

# Global request tracking for concurrent metrics
active_requests = 0
total_requests_processed = 0
server_start_time = time.time()

# Enhanced request statistics with mode breakdown
def _init_mode_stats():
    """Initialize statistics structure for a retrieval mode"""
    return {
        "total_requests": 0,
        "completed": 0,
        "failed": 0,
        "cache_hits": 0,
        "cache_misses": 0,
        "timings": {
            "total": 0.0,
            "search": 0.0,
            "content": 0.0,
            "llm": 0.0,
            "cache_write": 0.0
        },
        "errors_by_stage": {
            "search": 0,
            "content_processing": 0,
            "llm_generation": 0,
            "cache_write": 0,
            "unknown": 0
        }
    }

request_stats = {
    "by_mode": {
        "google_serper": _init_mode_stats(),
        "local": _init_mode_stats(),
        "unknown": _init_mode_stats()  # For validation errors and edge cases
    },
    "recent_errors": []  # Keep last 100 errors
}

def _record_success(mode: str, timings: dict, cached: bool):
    """Record a successful request"""
    # Handle invalid modes by defaulting to unknown
    if mode not in request_stats["by_mode"]:
        mode = "unknown"

    stats = request_stats["by_mode"][mode]
    stats["total_requests"] += 1
    stats["completed"] += 1
    if cached:
        stats["cache_hits"] += 1
    else:
        stats["cache_misses"] += 1

    # Update timings
    for stage, duration in timings.items():
        if stage in stats["timings"]:
            stats["timings"][stage] += duration

def _record_failure(mode: str, stage: str, error_details: dict):
    """Record a failed request with error details"""
    # Handle invalid modes by defaulting to unknown
    if mode not in request_stats["by_mode"]:
        mode = "unknown"

    stats = request_stats["by_mode"][mode]
    stats["total_requests"] += 1
    stats["failed"] += 1

    # Categorize error by stage
    if stage in stats["errors_by_stage"]:
        stats["errors_by_stage"][stage] += 1
    else:
        stats["errors_by_stage"]["unknown"] += 1

    # Add to recent errors (keep last 100)
    error_entry = {
        "timestamp": datetime.now().isoformat(),
        **error_details
    }
    request_stats["recent_errors"].append(error_entry)
    if len(request_stats["recent_errors"]) > 100:
        request_stats["recent_errors"].pop(0)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global shared_memory_bank
    logger.info("Starting Web Search Server...")
    
    try:
        # Validate required environment variables
        required_env_vars = {
            'WEBSEARCH_GOOGLE_SERPER_KEY': 'Google Serper API key for web search',
            'AZURE_OPENAI_API_KEY': 'Azure OpenAI API key for LLM summarization'
        }
        
        missing_vars = []
        for env_var, description in required_env_vars.items():
            if not os.environ.get(env_var):
                missing_vars.append(f"  {env_var}: {description}")
        
        if missing_vars:
            error_msg = "Missing required environment variables:\n" + "\n".join(missing_vars)
            error_msg += "\n\nPlease set these environment variables before starting the server."
            logger.error(error_msg)
            raise EnvironmentError(error_msg)
        
        logger.info("✓ All required environment variables are set")
        
        # Check Playwright browser installation
        try:
            async with async_playwright() as p:
                # Try to get browser executable path
                browser_path = p.chromium.executable_path
                if not os.path.exists(browser_path):
                    raise FileNotFoundError(f"Chromium executable not found at {browser_path}")
            logger.info("✓ Playwright browsers are installed and accessible")
        except Exception as e:
            error_msg = (
                f"Playwright browser check failed: {e}\n\n"
                "Please install Playwright browsers and dependencies by running:\n"
                "  playwright install-deps\n"
                "  playwright install\n\n"
                "Or if playwright is not in PATH:\n"
                "  python -m playwright install-deps\n"
                "  python -m playwright install\n\n"
                "Note: install-deps installs system dependencies (on Linux/Ubuntu),\n"
                "then install downloads the actual browser binaries.\n\n"
                "The server cannot function properly without browsers for web scraping."
            )
            logger.error(error_msg)
            raise EnvironmentError(error_msg)
        
        # Load configuration
        if not os.path.exists(CONFIG_FILE):
            raise FileNotFoundError(f"Configuration file not found: {CONFIG_FILE}")
            
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)
        
        # Create tool instance
        # Override cache directory to be in server folder
        config['search']['cache_dir'] = CACHE_DIR

        # Ensure llm config path is relative to server directory
        if 'llm' in config and 'config_path' in config['llm']:
            llm_config_path = config['llm']['config_path']
            if not os.path.isabs(llm_config_path):
                # Make relative paths relative to server.py location
                server_dir = os.path.dirname(__file__)
                config['llm']['config_path'] = os.path.join(server_dir, llm_config_path)

        shared_memory_bank = WebSearchAndSummarizeTool(
            config=config,
            top_k=config['search'].get('top_k', 5)
        )
        
        # Log cache statistics on startup
        try:
            cache_stats = shared_memory_bank.sqlite_cache.get_stats()
            logger.info(f"Web Search Tool initialized with cache dir: {CACHE_DIR}")
            logger.info(f"Cache startup stats: {cache_stats['entry_count']} entries, "
                       f"{cache_stats['total_size_bytes'] / 1024:.1f} KB, "
                       f"DB path: {cache_stats['db_path']}")
        except Exception as e:
            logger.warning(f"Could not retrieve cache startup stats: {e}")
            logger.info(f"Web Search Tool initialized with cache dir: {CACHE_DIR}")
        
        # Log current logging configuration
        current_log_level = os.getenv("VERL_LOGGING_LEVEL", "INFO")
        logger.info(f"📋 Server Configuration:")
        logger.info(f"  - Main logging level: {current_log_level}")
        logger.info(f"  - Memory bank tool level: {logging.getLogger('memory_bank_tool').level}")
        logger.info(f"  - LLM generator level: {logging.getLogger('llm_generator').level}")
        logger.info(f"  - HTTP logs level: {logging.getLogger('httpx').level} (suppressed)")
        logger.info(f"  - Server logs saved to: {LOG_FILE}")
        
        # Show if debug mode is enabled
        if current_log_level == "DEBUG":
            logger.info("🐛 DEBUG MODE ENABLED - Verbose logging active")
        else:
            logger.info(f"ℹ️  Standard logging mode - Set VERL_LOGGING_LEVEL=DEBUG for verbose output")
        
    except Exception as e:
        logger.error(f"Failed to initialize server: {e}")
        raise
    
    yield  # Server runs here
    
    # Cleanup
    logger.info("Shutting down Web Search Server...")
    if shared_memory_bank:
        await shared_memory_bank.close()

    # Clean up shared httpx resources
    await cleanup_shared_resources()
    logger.info("✓ Shared resources cleaned up")

app = FastAPI(
    title="Web Search and Summarization Server",
    description="A server that performs web search using Google Serper API and generates summaries using LLM",
    version="1.0.0",
    lifespan=lifespan
)

class SearchRequest(BaseModel):
    query: str = Field(..., description="The search query to process")
    retrieval_mode: str = Field(..., description="Retrieval mode: 'google_serper' or 'local'")
    top_k: Optional[int] = Field(None, description="Number of top results to process (overrides config)")
    local_database_address: Optional[str] = Field(None, description="Address of local database server (for local mode)")
    web_search_address: Optional[str] = Field(None, description="Legacy parameter name for local_database_address")
    llm_model: Optional[str] = Field(None, description="LLM model to use (e.g., 'gpt-4o-mini')")
    prompt_type: Optional[str] = Field(None, description="Prompt type: 'gpt4o' or 'mmsearch_r1'")
    use_jina: Optional[bool] = Field(None, description="Whether to use Jina for page processing")
    enable_thinking: Optional[bool] = Field(False, description="Enable Qwen3 thinking mode (default: False)")

@app.post("/search", response_class=PlainTextResponse)
async def search(request: SearchRequest):
    """
    Perform web search and generate summary
    """
    global active_requests, request_stats

    if not shared_memory_bank:
        raise HTTPException(status_code=503, detail="Error: Server not initialized")

    # Start timing and request tracking
    start_time = time.time()
    active_requests += 1
    request_id = f"req_{int(start_time * 1000) % 100000}"
    
    logger.info(f"[{request_id}] SEARCH_START: '{request.query[:50]}...' (active_requests: {active_requests})")

    try:
        # Validate retrieval mode
        if request.retrieval_mode not in ["google_serper", "local"]:
            raise HTTPException(status_code=400, detail=f"Error: Invalid retrieval_mode: {request.retrieval_mode}. Must be 'google_serper' or 'local'")

        # Validate local mode requirements
        if request.retrieval_mode == "local":
            local_addr = request.local_database_address or request.web_search_address
            if not local_addr:
                raise HTTPException(status_code=400, detail="Error: local_database_address is required when retrieval_mode is 'local'")

        # Prepare parameters with all overrides
        params_dict = {
            "query": request.query,
            "retrieval_mode": request.retrieval_mode
        }
        if request.top_k is not None:
            params_dict["top_k"] = request.top_k
        if request.local_database_address is not None:
            params_dict["local_database_address"] = request.local_database_address
        elif request.web_search_address is not None:  # Legacy support
            params_dict["local_database_address"] = request.web_search_address

        # Log parameter usage for debugging
        llm_model = request.llm_model or "gpt-4o-mini"  # Default will be applied in orchestrator
        prompt_type = request.prompt_type or "gpt4o"   # Default will be applied in orchestrator
        use_jina = request.use_jina if request.use_jina is not None else False  # Default will be applied in orchestrator
        enable_thinking = request.enable_thinking if request.enable_thinking is not None else False  # Default will be applied in orchestrator

        logger.info(f"[{request_id}] PARAMS: model={llm_model}, prompt={prompt_type}, jina={use_jina}, thinking={enable_thinking}")

        if request.llm_model is not None:
            params_dict["llm_model"] = request.llm_model
        if request.prompt_type is not None:
            params_dict["prompt_type"] = request.prompt_type
        if request.use_jina is not None:
            params_dict["use_jina"] = request.use_jina
        if request.enable_thinking is not None:
            params_dict["enable_thinking"] = request.enable_thinking
        params = json.dumps(params_dict)

        # Execute search with detailed timing
        search_start_time = time.time()
        summary_response, _, metrics = await shared_memory_bank.execute(
            "search_instance",
            params
        )
        search_end_time = time.time()

        # Check for errors - prevents training data contamination
        if summary_response.startswith("Error:"):
            raise HTTPException(status_code=500, detail=summary_response)

        # Calculate timing breakdown
        total_time = time.time() - start_time
        search_time = search_end_time - search_start_time
        overhead_time = total_time - search_time

        # Update statistics with mode tracking and stage breakdown
        cached = metrics.get("cached", False)
        timings = {
            "total": total_time,
            "search": search_time * 0.15,  # Approximate: 15% search
            "content": search_time * 0.47,  # 47% content processing
            "llm": search_time * 0.36,      # 36% LLM generation
            "cache_write": search_time * 0.02  # 2% cache write
        }
        _record_success(request.retrieval_mode, timings, cached)

        # Calculate hit rate for this mode
        mode_stats = request_stats["by_mode"][request.retrieval_mode]
        mode_total = mode_stats["cache_hits"] + mode_stats["cache_misses"]
        mode_hit_rate = (mode_stats["cache_hits"] / mode_total * 100) if mode_total > 0 else 0.0

        # Enhanced logging with mode and detailed timing breakdown
        status = "🚀 CACHED" if cached else "⚡ PROCESSED"
        logger.info(f"[{request_id}][{request.retrieval_mode}] {status}: Total: {total_time:.1f}s | "
                   f"Search: {timings['search']:.1f}s | Content: {timings['content']:.1f}s | "
                   f"LLM: {timings['llm']:.1f}s | Cache: {timings['cache_write']:.1f}s | "
                   f"Mode Hit Rate: {mode_hit_rate:.0f}% | Active: {active_requests}")

        # Log detailed stats every 10 requests with mode breakdown
        google_stats = request_stats["by_mode"]["google_serper"]
        local_stats = request_stats["by_mode"]["local"]

        total_completed = google_stats["completed"] + local_stats["completed"]
        total_failed = google_stats["failed"] + local_stats["failed"]

        if total_completed % 10 == 0:
            # Calculate per-mode hit rates
            google_total = google_stats["cache_hits"] + google_stats["cache_misses"]
            local_total = local_stats["cache_hits"] + local_stats["cache_misses"]
            google_hit_rate = (google_stats["cache_hits"] / google_total * 100) if google_total > 0 else 0.0
            local_hit_rate = (local_stats["cache_hits"] / local_total * 100) if local_total > 0 else 0.0

            # Calculate overall hit rate
            overall_hits = google_stats["cache_hits"] + local_stats["cache_hits"]
            overall_total = google_total + local_total
            overall_hit_rate = (overall_hits / overall_total * 100) if overall_total > 0 else 0.0

            # Calculate average processing time per mode
            google_avg_time = (google_stats["timings"]["total"] / google_stats["completed"]) if google_stats["completed"] > 0 else 0.0
            local_avg_time = (local_stats["timings"]["total"] / local_stats["completed"]) if local_stats["completed"] > 0 else 0.0

            logger.info(f"📊 [MILESTONE] {total_completed} requests completed | "
                       f"🎯 Overall Hit Rate: {overall_hit_rate:.0f}% | 🚨 Failed: {total_failed}")
            logger.info(f"  🌐 Google Serper: {google_stats['completed']} completed, "
                       f"{google_stats['failed']} failed, "
                       f"Hit Rate: {google_hit_rate:.0f}%, "
                       f"Avg Time: {google_avg_time:.1f}s")
            logger.info(f"  🏠 Local: {local_stats['completed']} completed, "
                       f"{local_stats['failed']} failed, "
                       f"Hit Rate: {local_hit_rate:.0f}%, "
                       f"Avg Time: {local_avg_time:.1f}s")

        return summary_response

    except HTTPException as e:
        total_time = time.time() - start_time

        # Determine failure stage based on error message
        error_msg = str(e.detail).lower()
        if "retrieval_mode" in error_msg or "local_database_address" in error_msg:
            stage = "search"  # Validation errors
        elif "cache" in error_msg or "database" in error_msg:
            stage = "cache_write"
        elif "llm" in error_msg or "generate" in error_msg:
            stage = "llm_generation"
        elif "content" in error_msg or "scraping" in error_msg or "fetch" in error_msg:
            stage = "content_processing"
        else:
            stage = "unknown"

        # Record failure with detailed context
        _record_failure(
            mode=request.retrieval_mode if hasattr(request, 'retrieval_mode') else "unknown",
            stage=stage,
            error_details={
                "request_id": request_id,
                "query": request.query[:100],
                "error_type": "HTTPException",
                "error_message": e.detail,
                "status_code": e.status_code,
                "duration_seconds": round(total_time, 2)
            }
        )

        logger.error(f"[{request_id}][{getattr(request, 'retrieval_mode', 'unknown')}] 🚨 FAILED at {stage}: {total_time:.1f}s | "
                    f"Error: {e.detail} | Active: {active_requests}")
        raise
    except Exception as e:
        total_time = time.time() - start_time

        # Generic errors are categorized as unknown stage
        _record_failure(
            mode=request.retrieval_mode if hasattr(request, 'retrieval_mode') else "unknown",
            stage="unknown",
            error_details={
                "request_id": request_id,
                "query": request.query[:100] if hasattr(request, 'query') else "N/A",
                "error_type": type(e).__name__,
                "error_message": str(e),
                "duration_seconds": round(total_time, 2)
            }
        )

        logger.error(f"[{request_id}][{getattr(request, 'retrieval_mode', 'unknown')}] ❌ ERROR: {total_time:.1f}s | "
                    f"{type(e).__name__}: {str(e)} | Active: {active_requests}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    finally:
        active_requests -= 1

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "web-search-server"
    }

@app.get("/stats")
def get_stats():
    """Get comprehensive server and cache statistics with mode breakdown"""
    global active_requests, request_stats

    if not shared_memory_bank:
        raise HTTPException(status_code=503, detail="Error: Server not initialized")

    try:
        # Get cache statistics from the tool
        cache_stats = shared_memory_bank.sqlite_cache.get_stats()

        # Get stats for each mode
        google_stats = request_stats["by_mode"]["google_serper"]
        local_stats = request_stats["by_mode"]["local"]

        # Calculate aggregated metrics
        total_completed = google_stats["completed"] + local_stats["completed"]
        total_failed = google_stats["failed"] + local_stats["failed"]
        total_hits = google_stats["cache_hits"] + local_stats["cache_hits"]
        total_misses = google_stats["cache_misses"] + local_stats["cache_misses"]
        total_cache_requests = total_hits + total_misses
        overall_hit_rate = (total_hits / total_cache_requests * 100) if total_cache_requests > 0 else 0.0

        # Calculate per-mode metrics
        def _format_mode_stats(mode_stats, mode_name):
            mode_total_requests = mode_stats["cache_hits"] + mode_stats["cache_misses"]
            mode_hit_rate = (mode_stats["cache_hits"] / mode_total_requests * 100) if mode_total_requests > 0 else 0.0

            # Calculate average times per stage
            avg_times = {}
            if mode_stats["completed"] > 0:
                for stage, total_time in mode_stats["timings"].items():
                    avg_times[f"avg_{stage}_seconds"] = round(total_time / mode_stats["completed"], 2)
            else:
                for stage in mode_stats["timings"].keys():
                    avg_times[f"avg_{stage}_seconds"] = 0.0

            # Calculate error rate
            total_errors = sum(mode_stats["errors_by_stage"].values())
            error_rate = (total_errors / mode_stats["total_requests"] * 100) if mode_stats["total_requests"] > 0 else 0.0

            return {
                "total_requests": mode_stats["total_requests"],
                "completed": mode_stats["completed"],
                "failed": mode_stats["failed"],
                "cache_hits": mode_stats["cache_hits"],
                "cache_misses": mode_stats["cache_misses"],
                "cache_hit_rate_percent": round(mode_hit_rate, 2),
                "error_rate_percent": round(error_rate, 2),
                **avg_times,
                "errors_by_stage": mode_stats["errors_by_stage"]
            }

        # Server uptime and processing rate
        server_uptime = time.time() - server_start_time
        total_processing_time = google_stats["timings"]["total"] + local_stats["timings"]["total"]
        avg_rps = (total_completed / total_processing_time) if total_processing_time > 0 else 0.0

        return {
            "server": {
                "active_requests": active_requests,
                "total_completed": total_completed,
                "total_failed": total_failed,
                "cache_hits": total_hits,
                "cache_misses": total_misses,
                "cache_hit_rate_percent": round(overall_hit_rate, 2),
                "uptime_seconds": round(server_uptime, 2),
                "avg_requests_per_second": round(avg_rps, 3),
                "status": "running"
            },
            "retrieval_modes": {
                "google_serper": _format_mode_stats(google_stats, "google_serper"),
                "local": _format_mode_stats(local_stats, "local")
            },
            "recent_errors": request_stats["recent_errors"][-10:],  # Last 10 errors
            "web_scraping": get_web_progress_stats(),
            "llm_generation": get_llm_progress_stats(),
            "cache_db": cache_stats,
            "resource_health": {
                "search_semaphore": {
                    "limit": shared_memory_bank.rate_config.search_semaphore_limit,
                    "available": shared_memory_bank.search_semaphore._value,
                    "in_use": shared_memory_bank.rate_config.search_semaphore_limit - shared_memory_bank.search_semaphore._value
                },
                "llm_semaphore": {
                    "limit": shared_memory_bank.rate_config.llm_semaphore_limit,
                    "available": shared_memory_bank.llm_semaphore._value,
                    "in_use": shared_memory_bank.rate_config.llm_semaphore_limit - shared_memory_bank.llm_semaphore._value
                },
                "page_semaphore": {
                    "limit": shared_memory_bank.rate_config.page_semaphore_limit,
                    "available": shared_memory_bank.page_semaphore._value,
                    "in_use": shared_memory_bank.rate_config.page_semaphore_limit - shared_memory_bank.page_semaphore._value
                },
                "browser_pool": {
                    "total_browsers": shared_memory_bank.browser_pool.pool_size,
                    "failed_browsers": len(shared_memory_bank.browser_pool.failed_browsers),
                    "healthy_browsers": shared_memory_bank.browser_pool.pool_size - len(shared_memory_bank.browser_pool.failed_browsers),
                    "failure_rate_percent": round((len(shared_memory_bank.browser_pool.failed_browsers) / shared_memory_bank.browser_pool.pool_size) * 100, 1)
                }
            }
        }
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail="Error: Failed to retrieve stats")

if __name__ == "__main__":
    import uvicorn
    
    # Default development server settings
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    
    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
