# Web Search Serper Server

A standalone FastAPI server that provides web search and summarization capabilities using Google Serper API and LLM-powered content analysis.

## Features

- **Dual Retrieval Modes**: Google Serper API for web search or local database retrieval
- **Content Scraping**: Fetches and processes web page content using Playwright with stealth mode and optional Jina Reader API
- **Multi-Model LLM Support**: Supports Azure OpenAI (GPT-4o, GPT-5) and vLLM-hosted models (Qwen3 series) with configurable prompting strategies
- **SQLite Caching**: Persistent caching system for improved performance
- **Browser Pool**: Managed pool of 64 browsers with health monitoring and auto-recovery
- **Async Processing**: Handles multiple concurrent requests efficiently with rate limiting
- **Progress Tracking**: Real-time progress monitoring for web scraping operations
- **RESTful API**: Simple HTTP API compatible with existing client code

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
playwright install-deps
playwright install
```

### 2. Environment Variables

Set the required environment variables:

```bash
export WEBSEARCH_GOOGLE_SERPER_KEY="your_serper_api_key"
export AZURE_OPENAI_API_KEY="your_azure_openai_key"
export JINA_API_KEY="your_jina_api_key"  # Optional: for Jina Reader API (get from https://jina.ai/)

# For Summarizer LLM (Qwen3 series hosted via SGLang)
export SUMMARIZER_API_KEY="your_summarizer_api_key"  # Optional: for SGLang deployments
export SUMMARIZER_BASE_URL="http://your-summarizer-server:8123/v1"  # Summarizer LLM endpoint
```

### 3. Configuration

The server uses two configuration files:

- `config.json` - Main configuration including tool schema and memory bank settings
- `llm_config.json` - LLM model configurations for different providers

These are pre-configured but can be modified as needed.

## Usage

### Starting the Server

```bash
# Set environment variables
export WEBSEARCH_GOOGLE_SERPER_KEY='your_serper_api_key'
export AZURE_OPENAI_API_KEY='your_azure_openai_key'
export JINA_API_KEY='your_jina_api_key'  # Optional: for Jina Reader API

# Start server
uvicorn server:app --host 0.0.0.0 --port 8000
```

### API Endpoints

#### POST /search
Perform web search and generate summary.

**Request:**
```json
{
  "query": "What are the latest AI developments?",
  "retrieval_mode": "google_serper",
  "top_k": 5,
  "local_database_address": "http://your-database-server:8001",
  "llm_model": "qwen3-32b",
  "prompt_type": "mmsearch_r1",
  "use_jina": true
}
```

**Parameters:**
- `query` (string, required): The search query to process
- `retrieval_mode` (string, required): Either "google_serper" or "local"
- `top_k` (integer, optional): Number of top results to process (overrides config default)
- `local_database_address` (string, optional): Address of local database server (required when retrieval_mode is "local")
- `web_search_address` (string, optional): Legacy parameter name for local_database_address
- `llm_model` (string, optional): LLM model to use - supports "gpt-4o", "gpt-4o-mini", "gpt-5", "qwen3-32b", "qwen3-8b", "qwen3-30b-a3b-instruct-2507", "qwen3-next-80b-a3b-instruct" (defaults to "gpt-4o-mini")
- `prompt_type` (string, optional): Prompting strategy - "gpt4o" for detailed expert analysis or "mmsearch_r1" for concise 5-sentence summaries (defaults to "gpt4o")
- `use_jina` (boolean, optional): Whether to use Jina Reader API for enhanced content processing (defaults to false)

**Response:** (Plain text summary)
```
Based on recent search results...
```

#### GET /health
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "service": "web-search-server"
}
```

#### GET /stats
Get comprehensive server, cache, and resource statistics.

**Response:**
```json
{
  "server": {
    "active_requests": 2,
    "total_completed": 150,
    "total_failed": 3,
    "cache_hits": 85,
    "cache_misses": 45,
    "cache_hit_rate_percent": 65.38,
    "uptime_seconds": 3600.5,
    "avg_requests_per_second": 0.042,
    "status": "running"
  },
  "web_scraping": {
    "active_fetches": 1,
    "completed_fetches": 120,
    "failed_fetches": 8,
    "bot_blocked_fetches": 2,
    "total_attempts": 130,
    "success_rate_percent": 92.31,
    "bot_detection_rate_percent": 1.54
  },
  "cache_db": {
    "entry_count": 42,
    "total_size_bytes": 1048576,
    "db_path": "./search_cache/search_cache.db"
  },
  "resource_health": {
    "search_semaphore": {
      "limit": 100,
      "available": 98,
      "in_use": 2
    },
    "llm_semaphore": {
      "limit": 1024,
      "available": 510,
      "in_use": 2
    },
    "page_semaphore": {
      "limit": 512,
      "available": 511,
      "in_use": 1
    },
    "browser_pool": {
      "total_browsers": 64,
      "failed_browsers": 1,
      "healthy_browsers": 63,
      "failure_rate_percent": 1.6
    }
  }
}
```

### Client Usage

Make HTTP POST requests directly to the server:

```python
import requests

def call_web_search_server(server_address, query, retrieval_mode="google_serper", top_k=None,
                         local_database_address=None, llm_model=None, prompt_type=None, use_jina=None):
    payload = {
        "query": query,
        "retrieval_mode": retrieval_mode
    }
    if top_k:
        payload["top_k"] = top_k
    if local_database_address:
        payload["local_database_address"] = local_database_address
    if llm_model:
        payload["llm_model"] = llm_model
    if prompt_type:
        payload["prompt_type"] = prompt_type
    if use_jina is not None:
        payload["use_jina"] = use_jina
    response = requests.post(f"http://{server_address}/search", json=payload, timeout=180)
    response.raise_for_status()
    return response.text  # Returns plain text summary

# Usage examples
server_address = "127.0.0.1:8000"

# Google Serper search (default with GPT-4o-mini)
summary = call_web_search_server(server_address, "Python async programming")
print(summary)

# Google Serper with Qwen3-32B model and enhanced processing
summary = call_web_search_server(server_address, "Python async programming",
                                llm_model="qwen3-32b", prompt_type="mmsearch_r1", use_jina=True)
print(summary)

# Google Serper with GPT-4o for detailed analysis
summary = call_web_search_server(server_address, "Python async programming",
                                llm_model="gpt-4o", prompt_type="gpt4o", top_k=5)
print(summary)

# Local database search with custom configuration
summary = call_web_search_server(server_address, "Python async programming",
                                retrieval_mode="local",
                                local_database_address="http://your-database-server:8001",
                                llm_model="qwen3-8b")
print(summary)
```

Or using curl:
```bash
# Google Serper search with default settings
curl -X POST "http://127.0.0.1:8000/search" \
     -H "Content-Type: application/json" \
     -d '{"query": "Python async programming", "retrieval_mode": "google_serper"}'

# Google Serper search with Qwen3 model and enhanced processing
curl -X POST "http://127.0.0.1:8000/search" \
     -H "Content-Type: application/json" \
     -d '{"query": "Python async programming", "retrieval_mode": "google_serper", "llm_model": "qwen3-32b", "prompt_type": "mmsearch_r1", "use_jina": true}'

# Local database search with custom model
curl -X POST "http://127.0.0.1:8000/search" \
     -H "Content-Type: application/json" \
     -d '{"query": "Python async programming", "retrieval_mode": "local", "local_database_address": "http://your-database-server:8001", "llm_model": "qwen3-8b"}'
```

## Migration from Original Code

This server replaces the original `call_web_search_google_serper_memory_bank` function:

**Before:**
```python
from verl.tools.websearch import call_web_search_google_serper_memory_bank

summary = await call_web_search_google_serper_memory_bank(query, total=3)
```

**After:**
```python
import requests

def call_web_search_server(server_address, query, retrieval_mode="google_serper"):
    payload = {
        "query": query,
        "retrieval_mode": retrieval_mode
    }
    response = requests.post(f"http://{server_address}/search", json=payload, timeout=180)
    response.raise_for_status()
    return response.text

summary = call_web_search_server("127.0.0.1:8000", query)
```

## Cache Management

- Cache is stored in `./search_cache/search_cache.db` (SQLite database)
- Cache keys include query normalization for better hit rates
- No automatic cache expiration (accumulates indefinitely)
- Cache statistics available via `/stats` endpoint

## Performance Notes

- Browser pool size: 64 concurrent browsers
- Page semaphore: 512 concurrent pages
- Search semaphore: 100 concurrent searches
- LLM semaphore: 1024 concurrent LLM calls
- Serper API rate limit: 100 req/s

## Troubleshooting

### Common Issues

1. **Browser initialization fails**: Ensure Playwright browsers are installed (`playwright install`)
2. **API key errors**: Check environment variables are set correctly
3. **Timeout errors**: Increase client timeout for complex searches (default 180s)
4. **Port conflicts**: Use different port via `PORT` environment variable

### Logs

Server logs include detailed progress tracking:
- `[SEARCH_START/DONE]` - Search API calls
- `[WEB_START/DONE]` - Web page fetching
- `[CACHE_HIT/MISS]` - Cache operations
- `[WEB_PROGRESS]` - Active request counters

## File Structure

```
web_search_server/
├── server.py                    # FastAPI server application
├── memory_bank_tool.py          # Core search and summarization tool
├── search_orchestrator.py       # Search coordination and flow control
├── search_providers.py          # Google Serper and local retrieval providers
├── content_processor.py         # Web scraping and content processing
├── browser_pool.py              # Browser pool management
├── browser_health_monitor.py    # Browser health monitoring
├── browser_recovery_manager.py  # Browser failure recovery
├── user_agent_generator.py      # User agent rotation for stealth
├── llm_generator.py             # LLM interface for summarization
├── cache.py                     # SQLite caching system
├── progress_tracker.py          # Request progress tracking
├── config.py                    # Configuration dataclasses
├── schemas.py                   # Pydantic schemas
├── config.json                  # Main configuration
├── llm_config.json              # LLM model configurations
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── __init__.py                  # Python package marker
├── .gitignore                   # Git ignore rules
└── search_cache/                # SQLite cache database (created at runtime)
```