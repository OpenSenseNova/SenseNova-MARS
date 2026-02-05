#!/usr/bin/env python3

import argparse
import asyncio
import base64
import io
import json
import math
import os
import re
import string
import time
from typing import Optional

import atexit
import hashlib
import sqlite3

import aiohttp
import yaml
from PIL import Image

# Required for URL fetching with JS rendering
from playwright.async_api import async_playwright


class FatalAPIError(Exception):
    """Raised when Serper or LLM judge has an error that should stop evaluation."""
    pass


class URLFetchError(Exception):
    """Raised when URL fetch fails with a retriable error (timeout, HTTP error, etc.)."""
    pass


class WebFetchStats:
    """Global tracker for web fetch statistics."""
    def __init__(self):
        self.total = 0
        self.successful = 0
        self.failed = 0
        self.skipped = 0  # Non-HTML, skip extensions
        self.errors_by_code = {}  # HTTP status code -> count

    def record_success(self):
        self.total += 1
        self.successful += 1

    def record_failure(self, error_msg: str = ""):
        self.total += 1
        self.failed += 1
        # Extract HTTP status code if present
        match = re.search(r'HTTP (\d+)', error_msg)
        if match:
            code = match.group(1)
            self.errors_by_code[code] = self.errors_by_code.get(code, 0) + 1

    def record_skip(self):
        self.total += 1
        self.skipped += 1

    def get_stats(self) -> dict:
        return {
            "total": self.total,
            "successful": self.successful,
            "failed": self.failed,
            "skipped": self.skipped,
            "success_rate": (self.successful / self.total * 100) if self.total > 0 else 0,
            "errors_by_code": self.errors_by_code,
        }

    def format_progress(self) -> str:
        if self.total == 0:
            return ""
        rate = self.successful / self.total * 100
        failed_str = f", {self.failed} failed" if self.failed > 0 else ""
        skipped_str = f", {self.skipped} skipped" if self.skipped > 0 else ""
        return f"Web: {self.successful}/{self.total} ({rate:.0f}%){failed_str}{skipped_str}"


# Global web fetch stats tracker
_web_fetch_stats = WebFetchStats()


class SearchCache:
    """Simple SQLite cache for search results."""

    def __init__(self, cache_dir: str):
        self.db_path = os.path.join(cache_dir, "search_cache.db")
        os.makedirs(cache_dir, exist_ok=True)
        self._init_db()
        self.hits = 0
        self.misses = 0
        # Print initial stats
        count = self._get_entry_count()
        print(f"Search cache: {self.db_path} ({count} entries)")
        # Register cleanup on exit
        atexit.register(self.close)

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
        finally:
            conn.close()

    def _get_entry_count(self) -> int:
        conn = sqlite3.connect(self.db_path)
        try:
            count = conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
        finally:
            conn.close()
        return count

    def _make_key(self, query: str, top_k: int, model: str) -> str:
        # Create normalized cache key
        normalized = re.sub(r'\s+', ' ', query.strip().lower())
        parts = [
            f"q={normalized}",
            f"k={top_k}",
            f"model={model}",
            f"prompt=mmsearch_r1",
            f"jina=0",
            f"think=0"
        ]
        return hashlib.md5("|".join(parts).encode("utf-8")).hexdigest()

    async def get(self, query: str, top_k: int, model: str) -> Optional[str]:
        key = self._make_key(query, top_k, model)
        conn = sqlite3.connect(self.db_path)
        try:
            row = conn.execute("SELECT value FROM cache WHERE key = ?", (key,)).fetchone()
        finally:
            conn.close()
        if row:
            self.hits += 1
            print(f"[CACHE HIT] {query[:50]}...")
            # Cache stores JSON {"summaries": "..."}, extract summaries field
            try:
                data = json.loads(row[0])
                if isinstance(data, dict) and "summaries" in data:
                    return data["summaries"]
                return row[0]  # Fallback: return as-is if not JSON format
            except json.JSONDecodeError:
                return row[0]  # Plain string, return as-is
        self.misses += 1
        return None

    async def set(self, query: str, top_k: int, model: str, value: str):
        # Store in JSON format
        key = self._make_key(query, top_k, model)
        data = json.dumps({"summaries": value})
        for attempt in range(5):
            conn = sqlite3.connect(self.db_path, timeout=5.0)
            try:
                conn.execute("INSERT OR REPLACE INTO cache (key, value) VALUES (?, ?)", (key, data))
                conn.commit()
                print(f"[CACHE STORE] {query[:50]}...")
                return
            except sqlite3.OperationalError as e:
                if "locked" in str(e).lower() and attempt < 4:
                    await asyncio.sleep(0.05 * (2 ** attempt))
                else:
                    raise
            finally:
                conn.close()

    def get_stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "total": total,
            "hit_rate": (self.hits / total * 100) if total > 0 else 0.0,
        }

    def close(self):
        """Checkpoint WAL to prevent corruption on shutdown."""
        try:
            conn = sqlite3.connect(self.db_path, timeout=10.0)
            try:
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            finally:
                conn.close()
        except Exception:
            pass  # Best effort on shutdown


def load_tool_config(tool_config_path: str) -> str:
    with open(tool_config_path) as f:
        config = yaml.safe_load(f)

    tools = config.get("tools", [])
    tool_definitions = []

    for tool in tools:
        schema = tool.get("tool_schema", {})
        tool_definitions.append(json.dumps(schema, ensure_ascii=False))

    print(f"  Loaded {len(tools)} tools")

    # Build # Tools section
    tool_def_str = "\n".join(tool_definitions)
    tools_section = f"""# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tool_def_str}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>"""

    return tools_section


# =============================================================================
# Image Processing
# =============================================================================

def round_by_factor(number: int, factor: int) -> int:
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    return math.floor(number / factor) * factor


def smart_resize(
    height: int,
    width: int,
    factor: int = 32,
    min_pixels: int = 65536,
    max_pixels: int = 8294400,
) -> tuple[int, int]:
    """Resize dimensions to be divisible by factor and within pixel limits."""
    if max(height, width) / min(height, width) > 200:
        raise ValueError("Aspect ratio too extreme")

    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))

    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(int(height / beta), factor)
        w_bar = floor_by_factor(int(width / beta), factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(int(height * beta), factor)
        w_bar = ceil_by_factor(int(width * beta), factor)

    return h_bar, w_bar


def process_image(
    image: Image.Image,
    min_pixels: int = 65536,
    max_pixels: int = 8294400,
    factor: int = 32,
    qwen_vl_processing: bool = True,
) -> Image.Image:
    """Process image with smart_resize (Qwen-VL style) or simple max_pixels resize."""
    if image.mode != "RGB":
        image = image.convert("RGB")

    width, height = image.size

    if qwen_vl_processing:
        # Qwen-VL style: align to factor and respect min/max pixels
        resized_height, resized_width = smart_resize(height, width, factor, min_pixels, max_pixels)
    else:
        # Simple resize: just ensure within max_pixels
        if width * height > max_pixels:
            scale = math.sqrt(max_pixels / (width * height))
            resized_width = int(width * scale)
            resized_height = int(height * scale)
        else:
            resized_width, resized_height = width, height

    if (resized_width, resized_height) != (width, height):
        image = image.resize((resized_width, resized_height), Image.BICUBIC)

    return image


def load_and_process_image(
    path: str,
    min_pixels: int = 65536,
    max_pixels: int = 8294400,
    factor: int = 32,
    qwen_vl_processing: bool = True,
) -> Image.Image:
    """Load image from path and process it."""
    img = Image.open(path)
    img.load()
    return process_image(img, min_pixels, max_pixels, factor, qwen_vl_processing)


def image_to_base64(img: Image.Image) -> tuple[str, str]:
    """Convert PIL Image to base64 string."""
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    base64_str = base64.standard_b64encode(buffer.getvalue()).decode("utf-8")
    return base64_str, "image/png"


def crop_image(
    image: Image.Image,
    bbox: list,
    coord_scale: float = 1000.0,
    min_pixels: int = 65536,
    max_pixels: int = 8294400,
    factor: int = 32,
    qwen_vl_processing: bool = True,
    padding: tuple = (0.0, 0.0),  # Aligned with training config (no padding)
) -> Image.Image:
    """Crop image using bbox [x1, y1, x2, y2] in 0-1000 coords.

    - Supports configurable padding (default: no padding)
    - Padding is capped at 600px
    - Uses smart_resize for output
    """
    img_w, img_h = image.size

    # Normalize to 0-1 range
    x1, y1, x2, y2 = [float(c) / coord_scale for c in bbox]

    # Clamp to valid range before padding
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(1, x2), min(1, y2)

    # Apply padding if specified (capped at 600px)
    if padding[0] > 0 or padding[1] > 0:
        padding_cap = (600.0 / img_w, 600.0 / img_h)
        actual_padding = (
            min(padding[0], padding_cap[0]),
            min(padding[1], padding_cap[1]),
        )
        x1 = max(0.0, x1 - actual_padding[0])
        y1 = max(0.0, y1 - actual_padding[1])
        x2 = min(1.0, x2 + actual_padding[0])
        y2 = min(1.0, y2 + actual_padding[1])

    crop_box = (int(x1 * img_w), int(y1 * img_h), int(x2 * img_w), int(y2 * img_h))
    cropped = image.crop(crop_box)

    if qwen_vl_processing:
        # Qwen-VL: ensure minimum size and apply smart_resize
        w, h = cropped.size
        if w < 28 or h < 28:
            cropped = cropped.resize((max(w, 28), max(h, 28)), Image.Resampling.LANCZOS)
        return process_image(cropped, min_pixels, max_pixels, factor, qwen_vl_processing)
    else:
        # No processing: just convert to RGB
        if cropped.mode != "RGB":
            cropped = cropped.convert("RGB")
        return cropped


# =============================================================================
# Answer Extraction (em_score_mcq)
# =============================================================================

def extract_mcq_answer(text: str) -> Optional[str]:
    """Extract MCQ answer (A, B, C, D) from model output."""
    if not text:
        return None

    # Preprocess
    text = text.rsplit("<|im_start|>assistant", 1)[-1]
    text = re.split(r'</think(?:ing)?>', text)[-1]

    # Try <answer> tags first
    matches = list(re.finditer(r'<answer>(.*?)</answer>', text, re.DOTALL))
    if matches:
        candidate = matches[-1].group(1).strip()
        # Single letter
        if re.match(r'^[A-Da-d]$', candidate):
            return candidate.upper()
        # Letter with punctuation
        punct = re.findall(r'(?:\(([A-D])\)|\[([A-D])\]|(?<![A-Za-z])([A-D])[.\)\]])', candidate, re.IGNORECASE)
        if punct:
            last = punct[-1]
            return (last[0] or last[1] or last[2]).upper()
        # Standalone letter
        standalone = re.findall(r'(?<![A-Za-z])([A-D])(?![A-Za-z])', candidate)
        if standalone:
            return standalone[-1].upper()
        return None

    # Try \boxed{} - same extraction logic as <answer> tags
    match = re.search(r'\\boxed\{([^}]+)\}', text, re.IGNORECASE)
    if match:
        boxed = match.group(1).strip()
        if re.search(r'[A-Da-d]', boxed):
            # Single letter
            if re.match(r'^[A-Da-d]$', boxed):
                return boxed.upper()
            # Letter with punctuation
            punct = re.findall(r'(?:\(([A-D])\)|\[([A-D])\]|(?<![A-Za-z])([A-D])[.\)\]])', boxed, re.IGNORECASE)
            if punct:
                last = punct[-1]
                return (last[0] or last[1] or last[2]).upper()
            # Standalone letter
            standalone = re.findall(r'(?<![A-Za-z])([A-D])(?![A-Za-z])', boxed)
            if standalone:
                return standalone[-1].upper()
            # v7: boxed triggered but no letter extracted -> return None (skip fallback)
            return None

    # Fallback patterns on full text (only when no <answer> tag and no \boxed{} with a-d)
    # "Answer: (A)"
    answer_matches = re.findall(r'Answer:\s*\(([A-D])\)', text, re.IGNORECASE)
    if answer_matches:
        return answer_matches[-1].upper()

    # "answer is (A)"
    phrase = re.findall(r'(?:correct answer is|answer is)[:\s]*\(([A-D])\)', text, re.IGNORECASE)
    if phrase:
        return phrase[-1].upper()

    # Bold **(A)**
    bold_segments = re.findall(r'\*\*[^*]+\*\*', text)
    for seg in reversed(bold_segments):
        m = re.search(r'\(([A-D])\)', seg, re.IGNORECASE)
        if m:
            return m.group(1).upper()

    # (A) in parentheses
    paren = re.findall(r'\(([A-D])\)', text, re.IGNORECASE)
    if paren:
        return paren[-1].upper()

    # A) or A] format
    bracket = re.findall(r'(?<![A-Za-z])([A-D])[\)\]]', text, re.IGNORECASE)
    if bracket:
        return bracket[-1].upper()

    # Standalone letter
    standalone = re.findall(r'(?<![A-Za-z])([A-D])(?![A-Za-z])', text)
    if standalone:
        return standalone[-1].upper()

    return None


def check_answer(extracted: Optional[str], ground_truth: list | str) -> bool:
    """Check MCQ answer against ground truth(s)."""
    if extracted is None:
        return False
    # Handle list of ground truths
    if isinstance(ground_truth, str):
        ground_truth = [ground_truth]
    extracted_upper = extracted.upper()
    for gt in ground_truth:
        if extracted_upper == gt.upper():
            return True
    return False


# =============================================================================
# LLM Judge (llm_score)
# =============================================================================

def normalize_answer(s: str) -> str:
    """Normalize answer for EM comparison."""
    def remove_articles(text):
        if text.strip().lower() in ["a"]:
            return text
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_check(prediction: str, golden_answers) -> bool:
    """Check if prediction matches any golden answer."""
    if prediction is None:
        return False
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    for golden_answer in golden_answers:
        if normalize_answer(golden_answer) == normalized_prediction:
            return True
    return False


def extract_answer_allow_no_tag(text: str) -> str:
    """Extract answer from <answer> tags, fallback to full response."""
    if not text:
        return ""
    # Preprocess: remove assistant prefix and thinking tags
    text = text.rsplit("<|im_start|>assistant", 1)[-1]
    text = re.split(r'</think(?:ing)?>', text)[-1]
    # Try <answer> tags first
    matches = list(re.finditer(r'<answer>(.*?)</answer>', text, re.DOTALL))
    if matches:
        return matches[-1].group(1).strip()
    # Fallback: return entire response (stripped at <|im_end|> if present)
    return text.split('<|im_end|>')[0].strip()


# System prompt for LLM judge (13 principles)
LLM_JUDGE_SYSTEM_PROMPT = """You are an AI assistant tasked with evaluating the correctness of model responses based on an image, question, and ground truth answer. Your judgment should follow these principles:

1. Consider the image, question, and ground truth answer holistically before evaluating the model's response.
2. Your decision should be strictly Yes or No, based on whether the model's response is factually accurate and aligns with the ground truth answer.
3. If the model response is a more specific form of the ground truth answer, it is correct.
4. If the model response includes all key information but adds minor details, it is correct as long as the extra details are factually correct.
5. If the model response contradicts, modifies, or omits critical parts of the answer, it is incorrect.
6. For numerical values, ensure correctness even when presented in different units.
7. For names, check for first and last name correctness. If the middle name is extra but correct, consider it correct.
8. For yes/no questions, the response must exactly match "Yes" or "No" to be correct.
9. If the judgment can be made based solely on the text, you may choose to ignore the input image, as some images may be unfamiliar to you and could affect your judgment. Refer to the image only when necessary to minimize misjudgment.
10. If there are multiple candidate answers, you can also evaluate the model's response against all of them. If the response aligns with at least one candidate according to the rules above, it should be considered correct.
11. For multiple choice questions (A, B, C, D), be more lenient. If the model provides the correct letter choice, even with additional text or formatting, consider it correct.
12. If the model's answer contains the correct choice letter (A, B, C, or D) anywhere in the response, and it's clear this is the intended answer, mark it as correct.
13. Ignore formatting issues like extra parentheses, brackets, or minor text variations as long as the core answer is correct.

Your output must be in the following format:
<judge>Yes/No</judge>
<reason>Explanation of why the answer is correct or incorrect.</reason>"""

LLM_JUDGE_USER_PROMPT = """Image, Question, and Model Response Evaluation
Question: {question}
Ground Truth Answer: {ground_truth}
Model Response: {model_response}

Evaluation Instructions
Evaluate whether the Model Response is correct based on the Image, Question and Ground Truth Answer. Follow the predefined judgment rules and provide a clear Yes/No answer along with a justification.

Output Format
<judge>Yes/No</judge>
<reason>Detailed reasoning following the evaluation principles.</reason>"""


def get_judge_client(judge_client_type: str, judge_base_url: str, judge_api_key: str):
    """Create OpenAI client for LLM judge with max_retries=5."""
    from openai import AzureOpenAI, OpenAI

    if judge_client_type == "azure":
        api_version = os.environ.get("AZURE_API_VERSION", "2025-01-01-preview")
        return AzureOpenAI(
            api_key=judge_api_key,
            azure_endpoint=judge_base_url,
            api_version=api_version,
            max_retries=5,
        )
    else:
        # OpenAI official API
        return OpenAI(
            api_key=judge_api_key,
            max_retries=5,
        )


async def llm_judge_score(question: str, model_answer: str, ground_truth: list | str, image_path: str, judge_client: str, judge_base_url: str, judge_api_key: str, judge_temperature: float = 0.0) -> float:
    """Use LLM to judge if answer is correct. Returns 1.0 or 0.0.

    Flow:
    1. Preprocess and extract answer (fallback to full response)
    2. Check EM first - skip LLM if exact match
    3. Call LLM judge only if EM fails
    4. Loop through ALL ground truths (return 1.0 on first match)
    5. Send IMAGE to vision model judge
    6. Use OpenAI SDK with max_retries=5
    """
    # Extract answer with preprocessing
    extracted_answer = extract_answer_allow_no_tag(model_answer)

    # EM check first - skip LLM if exact match
    if em_check(extracted_answer, ground_truth):
        return 1.0

    # Convert to list if string
    if isinstance(ground_truth, str):
        ground_truth = [ground_truth]

    # Prepare image content once (reused for all ground truths)
    image_content = None
    if image_path and os.path.exists(image_path):
        # Convert image to base64 (JPEG quality=85)
        # Apply smart_resize to avoid "image too large" errors
        img = Image.open(image_path)
        img = process_image(img, min_pixels=65536, max_pixels=8294400, factor=32, qwen_vl_processing=True)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        image_content = {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}

    # Get client with max_retries=5
    client = get_judge_client(judge_client, judge_base_url, judge_api_key)
    model = "gpt-4o-2024-11-20"

    llm_score = 0.0

    # Loop through ALL ground truths
    for gt in ground_truth:
        # Format prompt with this ground truth
        user_prompt_text = LLM_JUDGE_USER_PROMPT.format(
            question=question, ground_truth=gt, model_response=extracted_answer
        )

        # Build user content - include image if available
        if image_content:
            user_content = [
                {"type": "text", "text": user_prompt_text},
                image_content
            ]
        else:
            user_content = user_prompt_text

        messages = [
            {"role": "system", "content": LLM_JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        try:
            # Use OpenAI SDK (has built-in retry with max_retries=5)
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=judge_temperature,
                max_tokens=8192,
                timeout=300,  # 5 min timeout for judge
            )
            content = response.choices[0].message.content

            match = re.search(r'<judge>\s*(Yes|No)\s*</judge>', content, re.IGNORECASE | re.DOTALL)
            if match and match.group(1).lower() == "yes":
                return 1.0  # Early return on first match
        except Exception as e:
            # After 5 retries failed, this is a serious issue
            error_msg = f"[LLM JUDGE ERROR] API call failed after 5 retries: {type(e).__name__}: {str(e)}"
            print(f"[ERROR] {error_msg}", flush=True)
            raise FatalAPIError(error_msg) from None

    return llm_score  # 0.0 if no ground truth matched


# =============================================================================
# Data Loading
# =============================================================================

def load_datasets(config_path: str, data_root: str = "") -> tuple[list[dict], dict]:
    """Load datasets from JSON config. Returns (samples, dataset_configs)."""
    with open(config_path) as f:
        config = json.load(f)

    all_samples = []
    dataset_configs = {}

    for dataset_name, ds_config in config.items():
        root = ds_config.get("root", "")
        annotation = ds_config.get("annotation", "")

        if data_root and not os.path.isabs(root):
            root = os.path.join(data_root, root)
        if data_root and not os.path.isabs(annotation):
            annotation = os.path.join(data_root, annotation)

        if not os.path.exists(annotation):
            print(f"Warning: {annotation} not found, skipping {dataset_name}")
            continue

        # Load samples
        samples = []
        with open(annotation) as f:
            for idx, line in enumerate(f):
                raw = json.loads(line)
                prompt = raw.get("prompt", [])
                question = ""
                for msg in prompt:
                    if msg.get("role") == "user":
                        question = msg.get("content", "").replace("<image>", "").strip()
                        break

                reward_model = raw.get("reward_model", {})
                # Keep full ground_truth list - loops through all for LLM judge
                answer = reward_model.get("ground_truth", [""])
                images = raw.get("image", [])
                image_path = os.path.join(root, images[0]) if images else ""

                # Extract image search data if present (for data-driven image_search_tool)
                # Data is at top level in data.jsonl: image_search_title_list, image_search_thumbnail_list
                # Limit to 5 results (default)
                IMAGE_SEARCH_MAX_RESULTS = 5
                image_search_kwargs = {}
                if "image_search_title_list" in raw:
                    titles = raw["image_search_title_list"]
                    image_search_kwargs["image_search_title_list"] = titles[:IMAGE_SEARCH_MAX_RESULTS] if titles else None
                if "image_search_thumbnail_list" in raw:
                    thumbs = raw["image_search_thumbnail_list"]
                    image_search_kwargs["image_search_thumbnail_list"] = thumbs[:IMAGE_SEARCH_MAX_RESULTS] if thumbs else None

                sample = {
                    "id": f"{dataset_name}-{idx}",
                    "question": question,
                    "answer": answer,
                    "image_path": image_path,
                    "dataset": dataset_name,
                    "data_root": root,
                }
                if image_search_kwargs:
                    sample["image_search_data"] = image_search_kwargs

                samples.append(sample)

        all_samples.extend(samples)

        # Scoring methods: combine reward_fn and unused_reward_fn into single list
        reward_fn = ds_config.get("reward_fn", [])
        unused_reward_fn = ds_config.get("unused_reward_fn", [])
        score_methods = list(set(reward_fn + unused_reward_fn))

        if not score_methods:
            raise ValueError(f"Dataset '{dataset_name}' has no scoring methods (reward_fn or unused_reward_fn)")

        valid_methods = {"em_score_mcq", "llm_score"}
        invalid = set(score_methods) - valid_methods
        if invalid:
            raise ValueError(f"Dataset '{dataset_name}' has invalid scoring methods: {invalid}")

        # Get input_template config
        input_template = ds_config.get("input_template", {})
        template_args = input_template.get("arguments", {})
        system_prompt = template_args.get("system_prompt", "")
        format_instruction = template_args.get("format_instruction", "")

        dataset_configs[dataset_name] = {
            "score_methods": score_methods,
            "system_prompt": system_prompt,
            "format_instruction": format_instruction,
        }

        print(f"  Loaded {len(samples)} samples from {dataset_name}")
        print(f"    reward_fn: {reward_fn}")
        print(f"    unused_reward_fn: {unused_reward_fn}")

    return all_samples, dataset_configs


# =============================================================================
# API Calls
# =============================================================================

async def call_openai_api(
    messages: list[dict],
    model: str,
    base_url: str,
    api_key: str = "",
    **kwargs,
) -> dict:
    """Call OpenAI-compatible API."""
    # Build request
    # Note: Don't set temperature by default - let API use its default (1.0)
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": kwargs.get("max_tokens", 4096),
    }
    if "temperature" in kwargs:
        body["temperature"] = kwargs["temperature"]
    if "top_p" in kwargs:
        body["top_p"] = kwargs["top_p"]
    if "top_k" in kwargs:
        body["top_k"] = kwargs["top_k"]
    if "presence_penalty" in kwargs:
        body["presence_penalty"] = kwargs["presence_penalty"]
    if "repetition_penalty" in kwargs:
        body["repetition_penalty"] = kwargs["repetition_penalty"]
    if "seed" in kwargs and kwargs["seed"] is not None:
        body["seed"] = kwargs["seed"]
    # For Qwen3 models via vLLM/SGLang - control thinking mode
    if kwargs.get("extra_body"):
        body.update(kwargs["extra_body"])

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    url = f"{base_url.rstrip('/')}/v1/chat/completions"

    try:
        timeout = aiohttp.ClientTimeout(total=300)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, headers=headers, json=body) as resp:
                data = await resp.json()
                if resp.status != 200:
                    error = data.get('error', {}).get('message', f'HTTP {resp.status}')
                    # Truncate base64 from error
                    error = re.sub(r'data:image[^"]*', '[IMAGE]', str(error))[:500]
                    return {"content": "", "finish_reason": "error", "error": error}

                choice = data["choices"][0]
                return {
                    "content": choice["message"]["content"],
                    "finish_reason": choice.get("finish_reason", "stop"),
                    "error": None,
                }
    except Exception as e:
        return {"content": "", "finish_reason": "error", "error": str(e)}


async def call_gemini_api(
    messages: list[dict],
    model: str,
    base_url: str,
    api_key: str,
    **kwargs,
) -> dict:
    """Call Gemini API."""
    # Convert messages to Gemini format
    contents = []
    for msg in messages:
        role = "user" if msg["role"] == "user" else "model"
        parts = []
        content = msg.get("content", [])
        if isinstance(content, str):
            parts.append({"text": content})
        else:
            for item in content:
                if isinstance(item, str):
                    parts.append({"text": item})
                elif item.get("type") == "text":
                    parts.append({"text": item["text"]})
                elif item.get("type") == "image_url":
                    url = item["image_url"]["url"]
                    if url.startswith("data:"):
                        mime, data = url.split(";base64,", 1)
                        mime = mime.replace("data:", "")
                        parts.append({"inlineData": {"mimeType": mime, "data": data}})
        contents.append({"role": role, "parts": parts})

    body = {
        "contents": contents,
        "generationConfig": {
            "temperature": kwargs.get("temperature", 0.7),
            "maxOutputTokens": kwargs.get("max_tokens", 4096),
        }
    }

    url = f"{base_url.rstrip('/')}/models/{model}:generateContent"
    headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}

    try:
        timeout = aiohttp.ClientTimeout(total=300)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, headers=headers, json=body) as resp:
                data = await resp.json()
                if resp.status != 200:
                    error = data.get('error', {}).get('message', f'HTTP {resp.status}')
                    return {"content": "", "finish_reason": "error", "error": error}

                content = ""
                if data.get("candidates"):
                    parts = data["candidates"][0].get("content", {}).get("parts", [])
                    content = "".join(p.get("text", "") for p in parts)

                return {
                    "content": content,
                    "finish_reason": "stop",
                    "error": None,
                }
    except Exception as e:
        return {"content": "", "finish_reason": "error", "error": str(e)}


async def call_azure_api(
    messages: list[dict],
    model: str,
    base_url: str,
    api_key: str,
    **kwargs,
) -> dict:
    """Call Azure OpenAI API."""
    api_version = kwargs.get("azure_api_version", "2025-04-01-preview")

    body = {
        "model": model,
        "messages": messages,
        "temperature": kwargs.get("temperature", 0.7),
        "max_tokens": kwargs.get("max_tokens", 4096),
    }
    if "top_p" in kwargs:
        body["top_p"] = kwargs["top_p"]
    if "reasoning_effort" in kwargs:
        body["reasoning_effort"] = kwargs["reasoning_effort"]

    url = f"{base_url.rstrip('/')}/openai/deployments/{model}/chat/completions?api-version={api_version}"
    headers = {"api-key": api_key, "Content-Type": "application/json"}

    try:
        timeout = aiohttp.ClientTimeout(total=300)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, headers=headers, json=body) as resp:
                data = await resp.json()
                if resp.status != 200:
                    error = data.get('error', {}).get('message', f'HTTP {resp.status}')
                    error = re.sub(r'data:image[^"]*', '[IMAGE]', str(error))[:500]
                    return {"content": "", "finish_reason": "error", "error": error}

                choice = data["choices"][0]
                return {
                    "content": choice["message"]["content"],
                    "finish_reason": choice.get("finish_reason", "stop"),
                    "error": None,
                }
    except Exception as e:
        return {"content": "", "finish_reason": "error", "error": str(e)}


# =============================================================================
# Tool Execution
# =============================================================================

def parse_tool_call(text: str) -> Optional[dict]:
    """Parse tool call from model output."""
    match = re.search(r'<tool_call>\s*(.*?)\s*</tool_call>', text, re.DOTALL)
    if match:
        try:
            content = match.group(1).strip()
            return json.loads(content)
        except json.JSONDecodeError:
            return None
    return None


# Summarization prompt (mmsearch_r1 style - 5 sentence summary)
# Used for both per-URL summaries and final summary
SUMMARY_SYSTEM_PROMPT = """You are a helpful assistant. Your task is to summarize the main content of the given web page in no more than five sentences. Your summary should cover the overall key points of the page, not just parts related to the user's question.

If any part of the content is helpful for answering the user's question, be sure to include it clearly in the summary. Do not ignore relevant information, but also make sure the general structure and main ideas of the page are preserved. Your summary should be concise, factual, and informative."""

SUMMARY_USER_PROMPT = """Webpage Content (first {content_limit} characters) is: {content}
Question: {query}"""


# Skip extensions for non-HTML resources
# Aligned with server config.json excluded_extensions
SKIP_EXTENSIONS = ['.pdf', '.doc', '.docx', '.ppt', '.pptx', '.xls', '.xlsx', '.jpg', '.jpeg', '.png', '.gif']

# Global browser context for reuse
_playwright = None
_browser = None
_browser_context = None


async def _get_browser_context():
    """Get or create a shared browser context."""
    global _playwright, _browser, _browser_context
    if _browser_context is None:
        _playwright = await async_playwright().start()
        _browser = await _playwright.chromium.launch(headless=True)
        # Use Chrome on Windows user agent
        _browser_context = await _browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.6261.57 Safari/537.36"
        )
    return _browser_context


async def _cleanup_browser():
    """Cleanup browser resources."""
    global _playwright, _browser, _browser_context
    if _browser_context:
        await _browser_context.close()
        _browser_context = None
    if _browser:
        await _browser.close()
        _browser = None
    if _playwright:
        await _playwright.stop()
        _playwright = None


# Bot detection status codes
BOT_DETECTION_CODES = {403, 429, 406, 418, 421, 451}

# Tracking domains to block
TRACKING_DOMAINS = [
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


async def fetch_url_content(url: str, timeout_sec: int = 20) -> Optional[str]:
    """Fetch URL content using Playwright with JS rendering.

    Returns None for skip cases (non-HTML, skip extensions).
    Raises URLFetchError for retriable errors (timeout, HTTP error, etc.).
    """
    from bs4 import BeautifulSoup

    # Skip case: non-HTML resource extensions (don't retry)
    if any(url.lower().endswith(ext) for ext in SKIP_EXTENSIONS):
        return None

    page = None
    try:
        context = await _get_browser_context()
        page = await context.new_page()

        # Block unnecessary resources
        excluded_resource_types = ["image", "stylesheet", "font", "media", "websocket", "eventsource", "manifest"]
        async def resource_handler(route):
            if route.request.resource_type in excluded_resource_types:
                await route.abort()
            else:
                await route.continue_()
        await page.route("**/*", resource_handler)

        # Block tracking domains
        async def tracking_handler(route):
            await route.abort()
        for domain_pattern in TRACKING_DOMAINS:
            try:
                await page.route(domain_pattern, tracking_handler)
            except:
                pass

        # wait_until='domcontentloaded'
        try:
            response = await asyncio.wait_for(
                page.goto(url, timeout=timeout_sec * 1000, wait_until='domcontentloaded'),
                timeout=25.0
            )
        except asyncio.TimeoutError:
            raise URLFetchError(f"Timeout fetching {url}")

        if response is None:
            raise URLFetchError(f"No response from {url}")

        if response.status >= 400:
            raise URLFetchError(f"HTTP {response.status} from {url}")

        # Skip case: non-HTML content type (don't retry)
        content_type = response.headers.get('content-type', '')
        if 'text/html' not in content_type:
            return None

        try:
            html = await asyncio.wait_for(page.content(), timeout=15.0)
        except asyncio.TimeoutError:
            raise URLFetchError(f"Content timeout for {url}")

        # Parse with BeautifulSoup
        soup = BeautifulSoup(html, "lxml")
        for script in soup(["script", "style"]):
            script.decompose()
        raw_content = soup.get_text(separator='\n', strip=True)

        if not raw_content:
            raise URLFetchError(f"Empty content from {url}")

        return raw_content

    except URLFetchError:
        raise  # Re-raise URLFetchError for retry logic
    except Exception as e:
        raise URLFetchError(f"Error fetching {url}: {e}")
    finally:
        if page:
            try:
                await page.close()
            except:
                pass


def _clean_think_blocks(text: str) -> str:
    """Remove <think>...</think> blocks from response."""
    if not text:
        return text
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE).strip()


async def summarize_content(
    query: str,
    content: str,
    summarizer_base_url: str,
    summarizer_model: str,
    content_limit: int = 30000,
    max_retries: int = 5,  # Aligned with server config.json max_try_times=5
) -> Optional[str]:
    """Summarize content with LLM using mmsearch_r1 style prompt."""
    # Server has no minimum content check - send whatever content we have
    if not content:
        return None

    messages = [
        {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
        {"role": "user", "content": SUMMARY_USER_PROMPT.format(
            query=query,
            content=content[:content_limit],
            content_limit=content_limit,
        )},
    ]

    # Retry logic aligned with server's LLMGenerator (fixed 1s backoff)
    # Note: Server does NOT set temperature - uses API default (typically 1.0)
    # For Qwen3 models, disable thinking mode
    extra_body = None
    if "qwen3" in summarizer_model.lower():
        extra_body = {"chat_template_kwargs": {"enable_thinking": False}}

    for attempt in range(max_retries):
        try:
            result = await call_openai_api(
                messages=messages,
                model=summarizer_model,
                base_url=summarizer_base_url,
                api_key="",
                max_tokens=8192,
                extra_body=extra_body,
            )
            if result.get("error"):
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)  # Fixed 1s backoff like server
                    continue
                return None
            # Clean <think> blocks from response
            return _clean_think_blocks(result["content"])
        except Exception as e:
            if attempt < max_retries - 1:
                await asyncio.sleep(1)  # Fixed 1s backoff like server
            else:
                print(f"[SUMMARIZER ERROR] {e}")
                return None
    return None


async def call_text_search(
    query: str,
    serper_api_key: str,
    summarizer_base_url: str,
    summarizer_model: str,
    serper_semaphore: asyncio.Semaphore,
    serper_concurrency: int = 5,
    top_k: int = 3,
    content_limit: int = 30000,
    search_cache: Optional["SearchCache"] = None,
    max_serper_attempts: int = 3,
) -> str:
    """Call Google Serper API for text search, fetch each URL, summarize each, then generate final summary.

    Flow:
    1. Check cache (if enabled)
    2. Search via Serper to get URLs (with retry)
    3. For each URL: fetch content and summarize with LLM
    4. Generate final summary from all per-URL summaries
    5. Store in cache (if enabled)
    """
    # Step 0: Check cache
    if search_cache:
        cached = await search_cache.get(query, top_k, summarizer_model)
        if cached:
            return f"Found cached summary for query: {query}\n{cached}"

    # Step 1: Search via Serper (rate limited by semaphore)
    # Search API doesn't use "num" parameter - gets default results then slices to top_k
    # Retry logic with max_attempts=3 and exponential backoff
    url = "https://google.serper.dev/search"
    headers = {"X-API-KEY": serper_api_key, "Content-Type": "application/json"}
    payload = {"q": query}

    all_search_results = None
    search_results = None
    last_error = None

    for attempt in range(max_serper_attempts):
        async with serper_semaphore:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, headers=headers, json=payload) as resp:
                        if resp.status != 200:
                            error_text = await resp.text()
                            raise Exception(f"HTTP {resp.status}: {error_text}")
                        data = await resp.json()
                        organic_results = data.get("organic", [])
                        if not organic_results:
                            return f"No search results found for query: {query}"
                        # Restructure results
                        # all_search_results = ALL organic results (for display in final summary)
                        # search_results = top_k results (for URL fetching/summarization)
                        all_search_results = [
                            {'title': res.get('title', ''), 'snippet': res.get('snippet', ''), 'link': res.get('link', '')}
                            for res in organic_results
                        ]
                        search_results = all_search_results[:top_k]
                        break  # Success, exit retry loop
            except Exception as e:
                last_error = e
                if attempt < max_serper_attempts - 1:
                    # Exponential backoff: 1.0 * (attempt + 1) seconds
                    backoff = 1.0 * (attempt + 1)
                    print(f"[SERPER RETRY] Attempt {attempt + 1}/{max_serper_attempts} failed: {e}, retrying in {backoff}s...", flush=True)
                    await asyncio.sleep(backoff)

    # If all retries failed, raise error
    if search_results is None:
        error_msg = str(last_error) if last_error and str(last_error) else "Unknown error"
        raise FatalAPIError(f"[SERPER ERROR] Failed after {max_serper_attempts} attempts: {error_msg}") from last_error

    # Step 2: Fetch each URL and summarize in PARALLEL
    # 3 retries with exponential backoff
    async def fetch_and_summarize(item, max_retries: int = 3):
        item_url = item.get("link", "")
        if not item_url:
            return None

        for attempt in range(max_retries):
            try:
                content = await fetch_url_content(item_url)
                if not content:
                    # No content but no exception - don't retry (e.g., non-HTML, skip extensions)
                    _web_fetch_stats.record_skip()
                    return None
                summary = await summarize_content(
                    query=query,
                    content=content,
                    summarizer_base_url=summarizer_base_url,
                    summarizer_model=summarizer_model,
                    content_limit=content_limit,
                )
                _web_fetch_stats.record_success()
                return summary
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"[WEB] Failed to fetch {item_url} after {max_retries} attempts: {e}", flush=True)
                    _web_fetch_stats.record_failure(str(e))
                    return None
                # Exponential backoff: min(2^attempt, 5)
                backoff_time = min(2 ** attempt, 5)
                await asyncio.sleep(backoff_time)
        return None

    tasks = [fetch_and_summarize(item) for item in search_results]
    summaries = await asyncio.gather(*tasks, return_exceptions=True)
    # Filter out exceptions
    summaries = [s if not isinstance(s, Exception) else None for s in summaries]

    # Step 3: Format all content and generate final summary
    # Use ALL search results for Search Results section
    # But only top_k summaries in Web Page Summaries section
    all_content = f"Search Query: {query}\n\n"
    all_content += "--- Search Results ---\n"
    for i, result in enumerate(all_search_results, 1):
        all_content += f"[{i}] Title: {result.get('title', '')}\n"
        all_content += f"    Snippet: {result.get('snippet', 'No snippet')}\n"
        all_content += f"    Link: {result.get('link', 'No link')}\n\n"

    all_content += "--- Web Page Summaries ---\n"
    for summary in summaries:
        if summary:
            all_content += f"{summary}\n\n"

    # Generate final summary using same prompt as per-URL summaries
    final_summary = await summarize_content(
        query=query,
        content=all_content,
        summarizer_base_url=summarizer_base_url,
        summarizer_model=summarizer_model,
        content_limit=content_limit,
    )

    # Return error if final summary fails
    if not final_summary:
        return f"Error: Failed to generate final summary for query: {query}"

    result = final_summary

    # Store in cache
    if search_cache:
        await search_cache.set(query, top_k, summarizer_model, result)

    return f"Final summary generated for query: {query}\n{result}"


# =============================================================================
# Evaluation Functions
# =============================================================================

async def evaluate_direct(
    client: str,
    model: str,
    base_url: str,
    api_key: str,
    question: str,
    image_path: str,
    system_prompt: str = "",
    format_instruction: str = "",
    **kwargs,
) -> dict:
    """Direct mode - single turn, no tools."""
    # Process image
    img = load_and_process_image(image_path, kwargs.get("min_pixels", 65536),
                                  kwargs.get("max_pixels", 8294400), kwargs.get("factor", 32),
                                  kwargs.get("qwen_vl_processing", True))
    img_b64, mime = image_to_base64(img)

    # Build prompt with format instruction from dataset config
    prompt = question
    if format_instruction:
        prompt = f"{question}\n{format_instruction}"

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_b64}"}},
            {"type": "text", "text": prompt},
        ]
    })

    if client == "gemini":
        result = await call_gemini_api(messages, model, base_url, api_key, **kwargs)
    elif client == "azure":
        result = await call_azure_api(messages, model, base_url, api_key, **kwargs)
    else:
        result = await call_openai_api(messages, model, base_url, api_key, **kwargs)

    return {
        "output": result["content"],
        "input_messages": messages,
        "finish_reason": result["finish_reason"],
        "num_round": 1,
        "tool_calls": [],
        "error": result.get("error"),
        "saved_images": [],  # Direct mode doesn't save images for HTML
    }


async def evaluate_tool(
    client: str,
    model: str,
    base_url: str,
    api_key: str,
    question: str,
    image_path: str,
    tool_system_prompt: str,
    max_turns: int = 10,
    serper_api_key: str = "",
    image_search_data: Optional[dict] = None,
    sample_id: str = "",
    images_dir: str = "",
    **kwargs,
) -> dict:
    """Tool mode - multi-turn with tools.

    Args:
        image_search_data: Optional dict with pre-computed image search results:
            - image_search_title_list: List of result titles
            - image_search_thumbnail_list: List of thumbnail paths (relative to data root)
            - image_search_summary: Optional summary text
        sample_id: Unique identifier for this sample (for saving images)
        images_dir: Directory to save images for HTML viewer
    """
    # Track saved images for HTML viewer
    saved_images = []  # List of {"marker": "[IMAGE N]", "path": "filename.jpg", "type": "..."}
    image_counter = 0

    def save_image_for_html(img: Image.Image, img_type: str) -> str:
        """Save image and return filename."""
        nonlocal image_counter
        image_counter += 1
        if images_dir and sample_id:
            filename = f"{sample_id}_{img_type}_{image_counter}.jpg"
            filepath = os.path.join(images_dir, filename)
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.save(filepath, "JPEG", quality=85)
            return filename
        return ""

    # Load original image (for cropping)
    original_image = Image.open(image_path)
    original_image.load()
    if original_image.mode != "RGB":
        original_image = original_image.convert("RGB")

    # Process for initial display
    processed = process_image(original_image, kwargs.get("min_pixels", 65536),
                               kwargs.get("max_pixels", 8294400), kwargs.get("factor", 32),
                               kwargs.get("qwen_vl_processing", True))
    img_b64, mime = image_to_base64(processed)

    # Save input image for HTML
    input_img_path = save_image_for_html(processed, "input")
    if input_img_path:
        saved_images.append({"marker": f"[IMAGE {image_counter}]", "path": input_img_path, "type": "input"})

    # Build messages with tool system prompt from config
    messages = [
        {"role": "system", "content": tool_system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_b64}"}},
                {"type": "text", "text": question},
            ]
        }
    ]

    # Save initial messages for output
    initial_messages = [m.copy() for m in messages]

    tool_calls = []
    output_parts = []

    for turn in range(max_turns):
        if client == "gemini":
            result = await call_gemini_api(messages, model, base_url, api_key, **kwargs)
        elif client == "azure":
            result = await call_azure_api(messages, model, base_url, api_key, **kwargs)
        else:
            result = await call_openai_api(messages, model, base_url, api_key, **kwargs)

        output = result["content"]

        if result.get("error"):
            output_parts.append(output + "<|im_end|>")
            return {
                "output": "".join(output_parts),
                "input_messages": initial_messages,
                "finish_reason": "error",
                "num_round": turn + 1,
                "tool_calls": tool_calls,
                "error": result["error"],
                "saved_images": saved_images,
            }

        # Check for tool call first
        tool_call = parse_tool_call(output)
        if tool_call:
            tool_name = tool_call.get("name", "")
            args = tool_call.get("arguments", {})

            messages.append({"role": "assistant", "content": output})
            tool_response = ""

            if tool_name == "image_zoom_in_tool":
                bbox = args.get("bbox_2d") or args.get("bbox")
                if bbox and len(bbox) == 4:
                    cropped = crop_image(original_image, bbox,
                                         min_pixels=kwargs.get("min_pixels", 65536),
                                         max_pixels=kwargs.get("max_pixels", 8294400),
                                         factor=kwargs.get("factor", 32),
                                         qwen_vl_processing=kwargs.get("qwen_vl_processing", True))
                    crop_b64, crop_mime = image_to_base64(cropped)
                    # Save cropped image for HTML
                    crop_img_path = save_image_for_html(cropped, "zoom")
                    if crop_img_path:
                        saved_images.append({"marker": f"[IMAGE {image_counter}]", "path": crop_img_path, "type": "zoom"})
                    tool_calls.append({"name": tool_name, "bbox": bbox, "label": args.get("label", "")})
                    tool_response = f"<tool_response>\nHere is the zoomed image:[IMAGE {image_counter}]\n</tool_response>"
                    messages.append({
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "<tool_response>\nHere is the zoomed image:"},
                            {"type": "image_url", "image_url": {"url": f"data:{crop_mime};base64,{crop_b64}"}},
                            {"type": "text", "text": "\n</tool_response>"},
                        ]
                    })
                else:
                    tool_response = "<tool_response>\nError: Invalid bbox format.\n</tool_response>"
                    messages.append({"role": "user", "content": tool_response})

            elif tool_name == "text_search_tool":
                query = args.get("query", "")
                summarizer_base_url = kwargs.get("summarizer_base_url", "")
                summarizer_model = kwargs.get("summarizer_model", "")
                if query and serper_api_key and summarizer_base_url and summarizer_model:
                    serper_semaphore = kwargs.get("serper_semaphore")
                    search_cache = kwargs.get("search_cache")
                    serper_concurrency = kwargs.get("serper_concurrency", 5)
                    search_result = await call_text_search(
                        query=query,
                        serper_api_key=serper_api_key,
                        summarizer_base_url=summarizer_base_url,
                        summarizer_model=summarizer_model,
                        serper_semaphore=serper_semaphore,
                        serper_concurrency=serper_concurrency,
                        search_cache=search_cache,
                    )
                    tool_calls.append({"name": tool_name, "query": query})
                    tool_response = f"<tool_response>\n{search_result}\n</tool_response>"
                    messages.append({"role": "user", "content": tool_response})
                else:
                    tool_response = "<tool_response>\nError: Search not available.\n</tool_response>"
                    messages.append({"role": "user", "content": tool_response})

            elif tool_name == "image_search_tool":
                # Data-driven image search - results come from dataset
                tool_calls.append({"name": tool_name})
                if image_search_data:
                    title_list = image_search_data.get("image_search_title_list", [])
                    thumbnail_list = image_search_data.get("image_search_thumbnail_list", [])
                    data_root = kwargs.get("data_root", "")

                    if title_list and thumbnail_list:
                        # Build interleaved content with titles and thumbnails
                        content_parts = [{"type": "text", "text": "<tool_response>\nReverse Image Search Results:"}]
                        thumb_markers = []  # Track thumbnail markers for HTML
                        for i, title in enumerate(title_list):
                            content_parts.append({"type": "text", "text": f"\n\nTitle {i+1}: {title}\nThumbnail {i+1}: "})
                            if i < len(thumbnail_list):
                                thumb_path = thumbnail_list[i]
                                if data_root and not os.path.isabs(thumb_path):
                                    thumb_path = os.path.join(data_root, thumb_path)
                                if os.path.exists(thumb_path):
                                    thumb_img = load_and_process_image(thumb_path,
                                        kwargs.get("min_pixels", 65536),
                                        kwargs.get("max_pixels", 8294400),
                                        kwargs.get("factor", 32),
                                        kwargs.get("qwen_vl_processing", True))
                                    thumb_b64, thumb_mime = image_to_base64(thumb_img)
                                    content_parts.append({"type": "image_url", "image_url": {"url": f"data:{thumb_mime};base64,{thumb_b64}"}})
                                    # Save thumbnail for HTML
                                    thumb_img_path = save_image_for_html(thumb_img, "thumbnail")
                                    if thumb_img_path:
                                        thumb_markers.append(f"[IMAGE {image_counter}]")
                                        saved_images.append({"marker": f"[IMAGE {image_counter}]", "path": thumb_img_path, "type": "thumbnail"})
                        content_parts.append({"type": "text", "text": "\n</tool_response>"})
                        messages.append({"role": "user", "content": content_parts})
                        # Build tool_response with image markers for output
                        tool_response = "<tool_response>\nReverse Image Search Results:"
                        for i, title in enumerate(title_list):
                            tool_response += f"\n\nTitle {i+1}: {title}\nThumbnail {i+1}: "
                            if i < len(thumb_markers):
                                tool_response += thumb_markers[i]
                        tool_response += "\n</tool_response>"
                    else:
                        tool_response = "<tool_response>\nNo matching images were found.\n</tool_response>"
                        messages.append({"role": "user", "content": tool_response})
                else:
                    tool_response = "<tool_response>\nNo matching images were found.\n</tool_response>"
                    messages.append({"role": "user", "content": tool_response})

            else:
                tool_response = f"<tool_response>\nError: Unknown tool '{tool_name}'.\n</tool_response>"
                messages.append({"role": "user", "content": tool_response})

            # Add to output in conversation format
            output_parts.append(f"{output}<|im_end|><|im_start|>user\n{tool_response}<|im_end|>\n<|im_start|>assistant\n")
        else:
            # No tool call - terminate
            output_parts.append(output + "<|im_end|>")
            return {
                "output": "".join(output_parts),
                "input_messages": initial_messages,
                "finish_reason": "no_tool_calls",
                "num_round": turn + 1,
                "tool_calls": tool_calls,
                "error": None,
                "saved_images": saved_images,
            }

    # Max turns reached
    return {
        "output": "".join(output_parts),
        "input_messages": initial_messages,
        "finish_reason": "max_turns",
        "num_round": max_turns,
        "tool_calls": tool_calls,
        "error": None,
        "saved_images": saved_images,
    }


# =============================================================================
# Main Evaluation Loop
# =============================================================================

def messages_to_input_string(messages: list) -> str:
    """Convert messages to input string format like rollout data.

    Uses [IMAGE 1] marker for the first image (input image).
    """
    parts = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if isinstance(content, str):
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        else:
            # Handle multimodal content
            text_parts = []
            has_image = False
            for item in content:
                if isinstance(item, str):
                    text_parts.append(item)
                elif item.get("type") == "text":
                    text_parts.append(item["text"])
                elif item.get("type") == "image_url":
                    has_image = True

            content_str = "\n".join(text_parts)
            if has_image:
                # Use [IMAGE 1] for the input image
                content_str = "[IMAGE 1]\n" + content_str
            parts.append(f"<|im_start|>{role}\n{content_str}<|im_end|>")

    return "\n".join(parts) + "\n<|im_start|>assistant\n"


async def evaluate_sample(
    sample: dict,
    client: str,
    model: str,
    base_url: str,
    api_key: str,
    mode: str,
    dataset_configs: dict,
    semaphore: asyncio.Semaphore,
    **kwargs,
) -> dict:
    """Evaluate a single sample."""
    dataset_name = sample.get("dataset", "")
    ds_config = dataset_configs.get(dataset_name, {})
    score_methods = ds_config.get("score_methods", [])

    # Get prompts from dataset config
    system_prompt = ds_config.get("system_prompt", "")
    format_instruction = ds_config.get("format_instruction", "")

    async with semaphore:
        try:
            if mode == "direct":
                result = await evaluate_direct(
                    client, model, base_url, api_key,
                    sample["question"], sample["image_path"],
                    system_prompt=system_prompt,
                    format_instruction=format_instruction,
                    **kwargs
                )
            else:
                # For tool mode: combine base system_prompt from dataset + tools_section from --tool-config
                tools_section = kwargs.get("tools_section", "")
                if not system_prompt:
                    raise ValueError(f"system_prompt required in dataset config for tool mode (dataset: {dataset_name})")
                if not tools_section:
                    raise ValueError("tools_section required for tool mode (use --tool-config)")
                # Combine: base system_prompt + tools_section
                tool_system_prompt = system_prompt + "\n\n" + tools_section
                result = await evaluate_tool(
                    client, model, base_url, api_key,
                    sample["question"], sample["image_path"],
                    tool_system_prompt=tool_system_prompt,
                    image_search_data=sample.get("image_search_data"),
                    data_root=sample.get("data_root", ""),
                    sample_id=sample["id"],
                    **kwargs  # includes images_dir
                )

            # Compute scores dynamically based on score_methods from config
            scores = {}
            extracted = None

            for method in score_methods:
                if method == "em_score_mcq":
                    extracted = extract_mcq_answer(result["output"])
                    em_correct = check_answer(extracted, sample["answer"])
                    scores[method] = 1.0 if em_correct else 0.0
                elif method == "llm_score":
                    judge_client = kwargs.get("judge_client", "azure")
                    judge_base_url = kwargs.get("judge_base_url", "")
                    judge_api_key = kwargs.get("judge_api_key", "")
                    judge_temperature = kwargs.get("judge_temperature", 0.0)
                    scores[method] = await llm_judge_score(
                        sample["question"], result["output"], sample["answer"],
                        sample["image_path"],  # Send image to vision model judge
                        judge_client, judge_base_url, judge_api_key, judge_temperature
                    )
                else:
                    # Unknown method - skip or set to None
                    scores[method] = None

            # Convert messages to input string format
            input_str = messages_to_input_string(result["input_messages"])

            result_dict = {
                "sample_id": sample["id"],
                "dataset": dataset_name,
                "input": input_str,
                "output": result["output"],
                "gts": sample["answer"],  # answer is already a list
                "finish_reason": result["finish_reason"],
                "num_round": result["num_round"],
                "saved_images": result.get("saved_images", []),  # For HTML generation only
            }
            # Add all scores with their method names
            result_dict.update(scores)
            return result_dict

        except FatalAPIError:
            raise  # Re-raise to crash on SERPER/judge errors
        except Exception as e:
            print(f"[ERROR] {sample['id']}: {e}")
            error_dict = {
                "sample_id": sample["id"],
                "dataset": dataset_name,
                "input": "",
                "output": "",
                "gts": sample["answer"],  # answer is already a list
                "finish_reason": "error",
                "num_round": 0,
                "error": str(e),
            }
            # Add all score methods as 0.0 for errors
            for method in score_methods:
                error_dict[method] = 0.0
            return error_dict


async def run_evaluation(
    samples: list[dict],
    dataset_configs: dict,
    client: str,
    model: str,
    base_url: str,
    api_key: str,
    mode: str,
    max_concurrent: int = 4,
    output_dir: str = "",
    **kwargs,
) -> tuple[list[dict], dict]:
    """Run evaluation on all samples."""
    datasets = set(s.get("dataset", "") for s in samples)

    # Resume: load existing results and skip completed samples
    completed_ids = set()
    existing_results = []
    results_file = os.path.join(output_dir, "results.jsonl") if output_dir else ""
    if results_file and os.path.exists(results_file):
        with open(results_file) as f:
            for line in f:
                r = json.loads(line)
                existing_results.append(r)
                completed_ids.add(r.get("sample_id"))
        print(f"Resuming: loaded {len(existing_results)} completed results, skipping {len(completed_ids)} samples")
        samples = [s for s in samples if s["id"] not in completed_ids]

    print(f"Evaluating {len(samples)} samples across {len(datasets)} datasets")
    print(f"Client: {client}, Model: {model}")
    print(f"Mode: {mode}")
    print()

    # Incremental stats tracking (avoid recomputing from full results list)
    dataset_stats = {ds: {"total": 0, "errors": 0, "em_correct": 0, "em_total": 0, "llm_correct": 0, "llm_total": 0}
                     for ds in datasets}
    # Initialize with existing results
    for r in existing_results:
        ds = r.get("dataset", "")
        if ds in dataset_stats:
            dataset_stats[ds]["total"] += 1
            if r.get("error"):
                dataset_stats[ds]["errors"] += 1
            if r.get("em_score_mcq") is not None:
                dataset_stats[ds]["em_total"] += 1
                if r["em_score_mcq"] > 0.5:
                    dataset_stats[ds]["em_correct"] += 1
            if r.get("llm_score") is not None:
                dataset_stats[ds]["llm_total"] += 1
                if r["llm_score"] > 0.5:
                    dataset_stats[ds]["llm_correct"] += 1

    if not samples:
        print("All samples already completed!")
        # Generate HTML from existing results (note: saved_images not in JSONL, so images won't be inlined)
        if output_dir and existing_results:
            html_path = os.path.join(output_dir, "results.html")
            generate_html(existing_results, html_path, images_subdir="images")
            print(f"Generated HTML viewer: {html_path}", flush=True)
        # Return stats from existing results
        dataset_results = {}
        for dataset_name in sorted(datasets):
            st = dataset_stats[dataset_name]
            ds_stats = {"total": st["total"], "errors": st["errors"]}
            if st["em_total"] > 0:
                ds_stats["em_correct"] = st["em_correct"]
                ds_stats["em_total"] = st["em_total"]
                ds_stats["em_accuracy"] = st["em_correct"] / st["em_total"]
            if st["llm_total"] > 0:
                ds_stats["llm_correct"] = st["llm_correct"]
                ds_stats["llm_total"] = st["llm_total"]
                ds_stats["llm_accuracy"] = st["llm_correct"] / st["llm_total"]
            dataset_results[dataset_name] = ds_stats
        return dataset_results

    semaphore = asyncio.Semaphore(max_concurrent)

    # Create serper semaphore for rate limiting (must be inside event loop)
    serper_concurrency = kwargs.pop("serper_concurrency", 5)
    serper_semaphore = asyncio.Semaphore(serper_concurrency)
    kwargs["serper_semaphore"] = serper_semaphore
    kwargs["serper_concurrency"] = serper_concurrency

    # Create images directory for HTML viewer
    images_dir = ""
    if output_dir:
        images_dir = os.path.join(output_dir, "images")
        os.makedirs(images_dir, exist_ok=True)
    kwargs["images_dir"] = images_dir

    start_time = time.time()

    tasks = [
        evaluate_sample(s, client, model, base_url, api_key, mode, dataset_configs, semaphore, **kwargs)
        for s in samples
    ]

    num_existing = len(existing_results)
    last_log_time = time.time()

    # Prepare output file for incremental saves
    results_file_handle = None
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        # Append mode if resuming, write mode if fresh start
        mode = "a" if existing_results else "w"
        results_file_handle = open(os.path.join(output_dir, "results.jsonl"), mode)

    # Collect results for HTML generation (includes saved_images which is not in JSONL)
    all_results_for_html = []

    try:
        for i, task in enumerate(asyncio.as_completed(tasks)):
            try:
                result = await task
            except FatalAPIError as e:
                print(f"\n{'='*70}", flush=True)
                print(f"FATAL ERROR: {e}", flush=True)
                print(f"{'='*70}\n", flush=True)
                raise

            # Update incremental stats
            ds = result.get("dataset", "")
            if ds in dataset_stats:
                dataset_stats[ds]["total"] += 1
                if result.get("error"):
                    dataset_stats[ds]["errors"] += 1
                if result.get("em_score_mcq") is not None:
                    dataset_stats[ds]["em_total"] += 1
                    if result["em_score_mcq"] > 0.5:
                        dataset_stats[ds]["em_correct"] += 1
                if result.get("llm_score") is not None:
                    dataset_stats[ds]["llm_total"] += 1
                    if result["llm_score"] > 0.5:
                        dataset_stats[ds]["llm_correct"] += 1

            # Save immediately (skip keys not suitable for jsonl)
            if results_file_handle:
                jsonl_skip_keys = {"saved_images", "tool_calls"}
                jsonl_result = {k: v for k, v in result.items() if k not in jsonl_skip_keys}
                results_file_handle.write(json.dumps(jsonl_result, ensure_ascii=False) + "\n")
                results_file_handle.flush()

            # Collect for HTML generation (keeps saved_images)
            all_results_for_html.append(result)

            now = time.time()
            total_done = num_existing + i + 1
            if (i + 1) % 10 == 0 or (i + 1) == len(tasks) or (now - last_log_time > 30):
                elapsed = now - start_time
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (len(tasks) - i - 1) / rate if rate > 0 else 0

                # Use incremental stats instead of recomputing
                ds_stats_strs = []
                for ds_name in sorted(datasets):
                    st = dataset_stats[ds_name]
                    if st["em_total"] > 0:
                        ds_acc = 100 * st["em_correct"] / st["em_total"]
                        ds_stats_strs.append(f"{ds_name}(em): {st['em_correct']}/{st['em_total']} ({ds_acc:.1f}%)")
                    if st["llm_total"] > 0:
                        ds_acc = 100 * st["llm_correct"] / st["llm_total"]
                        ds_stats_strs.append(f"{ds_name}(llm): {st['llm_correct']}/{st['llm_total']} ({ds_acc:.1f}%)")

                # Cache stats
                cache_info = ""
                search_cache = kwargs.get("search_cache")
                if search_cache:
                    cs = search_cache.get_stats()
                    cache_info = f" | Cache: {cs['hits']}/{cs['total']} ({cs['hit_rate']:.0f}%)"

                # Web fetch stats
                web_stats = _web_fetch_stats.format_progress()
                web_info = f" | {web_stats}" if web_stats else ""

                print(f"Progress: {total_done}/{num_existing + len(tasks)} | Elapsed: {elapsed:.0f}s | {rate:.1f} samples/s | ETA: {eta:.0f}s{cache_info}{web_info}", flush=True)
                if ds_stats_strs:
                    print(f"  {' | '.join(ds_stats_strs)}", flush=True)
                last_log_time = now
    finally:
        if results_file_handle:
            results_file_handle.close()

    # Generate HTML viewer with all results (new + existing)
    # Note: existing_results from resume don't have saved_images (not in JSONL),
    # so their [IMAGE N] markers won't be replaced with actual images in HTML
    if output_dir and (all_results_for_html or existing_results):
        html_results = existing_results + all_results_for_html
        html_path = os.path.join(output_dir, "results.html")
        generate_html(html_results, html_path, images_subdir="images")
        print(f"Generated HTML viewer: {html_path}", flush=True)

    elapsed = time.time() - start_time

    # Per-dataset results (use incremental stats - no recomputation needed)
    print(flush=True)
    print("=" * 70, flush=True)
    print("Results by Dataset:", flush=True)
    print("=" * 70, flush=True)

    dataset_results = {}
    total_samples = 0
    for dataset_name in sorted(datasets):
        st = dataset_stats[dataset_name]
        total_samples += st["total"]

        ds_stats = {"total": st["total"], "errors": st["errors"]}
        if st["em_total"] > 0:
            ds_stats["em_correct"] = st["em_correct"]
            ds_stats["em_total"] = st["em_total"]
            ds_stats["em_accuracy"] = st["em_correct"] / st["em_total"]
        if st["llm_total"] > 0:
            ds_stats["llm_correct"] = st["llm_correct"]
            ds_stats["llm_total"] = st["llm_total"]
            ds_stats["llm_accuracy"] = st["llm_correct"] / st["llm_total"]

        dataset_results[dataset_name] = ds_stats

        print(f"\n{dataset_name} ({st['total']} samples):", flush=True)
        if "em_accuracy" in ds_stats:
            print(f"  em_score_mcq:            {ds_stats['em_accuracy']:.4f} ({ds_stats['em_correct']}/{ds_stats['em_total']})", flush=True)
        if "llm_accuracy" in ds_stats:
            print(f"  llm_score_allow_no_answer:  {ds_stats['llm_accuracy']:.4f} ({ds_stats['llm_correct']}/{ds_stats['llm_total']})", flush=True)
        if ds_stats["errors"] > 0:
            print(f"  errors: {ds_stats['errors']}", flush=True)

    print(flush=True)
    if total_samples > 0:
        print(f"Time: {elapsed:.1f}s ({elapsed/total_samples:.2f}s per sample)", flush=True)
    else:
        print(f"Time: {elapsed:.1f}s (no samples processed)", flush=True)

    # Web fetch stats summary
    ws = _web_fetch_stats.get_stats()
    if ws["total"] > 0:
        print(f"\nWeb Fetch Stats: {ws['successful']}/{ws['total']} ({ws['success_rate']:.1f}%) successful", flush=True)
        if ws["failed"] > 0:
            print(f"  Failed: {ws['failed']}", flush=True)
            if ws["errors_by_code"]:
                code_str = ", ".join(f"HTTP {code}: {cnt}" for code, cnt in sorted(ws["errors_by_code"].items()))
                print(f"  Error codes: {code_str}", flush=True)
        if ws["skipped"] > 0:
            print(f"  Skipped (non-HTML): {ws['skipped']}", flush=True)

    print("=" * 70, flush=True)

    # Cleanup browser in the SAME event loop (critical - Playwright hangs if cleaned up from different loop)
    await _cleanup_browser()

    # Return dataset_results only - results are already saved incrementally to JSONL
    return dataset_results


def generate_html(results: list[dict], output_path: str, images_subdir: str = "images"):
    """Generate HTML viewer for results with actual images displayed.

    Args:
        results: List of result dicts, each may contain 'saved_images' with image paths
        output_path: Path to write HTML file
        images_subdir: Subdirectory containing images (relative to HTML file)
    """
    import html as html_lib

    def highlight_tags_with_images(text: str, saved_images: list[dict]) -> str:
        """Add syntax highlighting for special tags and replace [IMAGE N] with actual images."""
        text = html_lib.escape(text)

        # Build marker -> img tag mapping
        for img_info in saved_images:
            marker = img_info.get("marker", "")
            path = img_info.get("path", "")
            img_type = img_info.get("type", "")
            if marker and path:
                escaped_marker = html_lib.escape(marker)
                # Create img tag with relative path
                img_tag = f'<br><img src="{images_subdir}/{path}" class="inline-image" title="{img_type}"><br>'
                text = text.replace(escaped_marker, img_tag)

        text = text.replace("\n", "<br>")
        # Highlight tags
        tag_styles = [
            (r"&lt;thinking&gt;", '<span class="tag-think">&lt;thinking&gt;</span>'),
            (r"&lt;/thinking&gt;", '<span class="tag-think">&lt;/thinking&gt;</span>'),
            (r"&lt;tool_call&gt;", '<span class="tag-tool">&lt;tool_call&gt;</span>'),
            (r"&lt;/tool_call&gt;", '<span class="tag-tool">&lt;/tool_call&gt;</span>'),
            (r"&lt;answer&gt;", '<span class="tag-answer">&lt;answer&gt;</span>'),
            (r"&lt;/answer&gt;", '<span class="tag-answer">&lt;/answer&gt;</span>'),
            (r"&lt;tool_response&gt;", '<span class="tag-response">&lt;tool_response&gt;</span>'),
            (r"&lt;/tool_response&gt;", '<span class="tag-response">&lt;/tool_response&gt;</span>'),
        ]
        for pattern, replacement in tag_styles:
            text = text.replace(pattern, replacement)
        return text

    html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Eval Results Viewer</title>
    <style>
        :root {
            --bg-primary: #1e1e2e; --bg-secondary: #282839; --bg-tertiary: #313244;
            --text-primary: #cdd6f4; --text-secondary: #a6adc8;
            --accent: #cba6f7; --green: #a6e3a1; --red: #f38ba8; --yellow: #f9e2af; --blue: #89b4fa;
            --border: #45475a;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: 'SF Mono', Consolas, monospace; background: var(--bg-primary); color: var(--text-primary); line-height: 1.6; padding: 20px; }
        .header { background: var(--bg-secondary); padding: 20px; border-radius: 12px; margin-bottom: 20px; border: 1px solid var(--border); }
        .header h1 { color: var(--accent); margin-bottom: 10px; font-size: 1.5em; }
        .stats { display: flex; gap: 20px; flex-wrap: wrap; }
        .stat { background: var(--bg-tertiary); padding: 8px 16px; border-radius: 8px; font-size: 0.9em; }
        .stat-label { color: var(--text-secondary); }
        .stat-value { color: var(--accent); font-weight: bold; }
        .controls { background: var(--bg-secondary); padding: 15px; border-radius: 12px; margin-bottom: 20px; display: flex; gap: 10px; flex-wrap: wrap; border: 1px solid var(--border); }
        .controls input { flex: 1; min-width: 200px; padding: 10px 15px; border: 1px solid var(--border); border-radius: 8px; background: var(--bg-tertiary); color: var(--text-primary); font-family: inherit; }
        .controls button { padding: 10px 20px; border: none; border-radius: 8px; background: var(--accent); color: var(--bg-primary); cursor: pointer; font-weight: bold; }
        .sample { background: var(--bg-secondary); border-radius: 12px; margin-bottom: 15px; overflow: hidden; border: 1px solid var(--border); }
        .sample-header { background: var(--bg-tertiary); padding: 15px 20px; cursor: pointer; display: flex; align-items: center; gap: 15px; border-bottom: 1px solid var(--border); }
        .sample-header:hover { background: #3b3b4f; }
        .toggle { color: var(--accent); transition: transform 0.2s; }
        .sample.collapsed .toggle { transform: rotate(-90deg); }
        .sample-title { flex: 1; font-weight: bold; }
        .sample-badges { display: flex; gap: 10px; }
        .badge { padding: 4px 12px; border-radius: 20px; font-size: 0.8em; font-weight: bold; }
        .badge-positive { background: var(--green); color: var(--bg-primary); }
        .badge-negative { background: var(--red); color: var(--bg-primary); }
        .badge-neutral { background: var(--text-secondary); color: var(--bg-primary); }
        .sample-content { padding: 20px; }
        .sample.collapsed .sample-content { display: none; }
        .meta-row { display: flex; gap: 15px; margin-bottom: 15px; flex-wrap: wrap; }
        .meta-item { background: var(--bg-tertiary); padding: 8px 12px; border-radius: 8px; font-size: 0.85em; }
        .meta-label { color: var(--text-secondary); }
        .meta-value { color: var(--green); }
        .conversation { display: flex; flex-direction: column; gap: 15px; }
        .turn { border-radius: 12px; overflow: hidden; border: 1px solid var(--border); }
        .turn-header { padding: 10px 15px; font-weight: bold; font-size: 0.85em; text-transform: uppercase; }
        .turn.user .turn-header { background: #2a4a6a; color: var(--blue); }
        .turn.assistant .turn-header { background: #2a4a3a; color: var(--green); }
        .turn-content { padding: 15px; background: var(--bg-tertiary); white-space: pre-wrap; word-wrap: break-word; font-size: 0.9em; max-height: 800px; overflow-y: auto; }
        .tag-think { color: var(--yellow); }
        .tag-tool { color: var(--accent); }
        .tag-answer { color: var(--green); }
        .tag-response { color: var(--blue); }
        .inline-image { max-width: 400px; max-height: 300px; border-radius: 8px; margin: 8px 0; border: 2px solid var(--border); cursor: pointer; transition: transform 0.2s; }
        .inline-image:hover { transform: scale(1.02); border-color: var(--accent); }
    </style>
</head>
<body>
    <div class="header">
        <h1>Eval Results Viewer</h1>
        <div class="stats">
            <div class="stat"><span class="stat-label">Samples:</span> <span class="stat-value">''' + str(len(results)) + '''</span></div>
            <div class="stat"><span class="stat-label">Generated:</span> <span class="stat-value">''' + time.strftime("%Y-%m-%d %H:%M:%S") + '''</span></div>
        </div>
    </div>
    <div class="controls">
        <input type="text" id="search" placeholder="Search samples..." onkeyup="filterSamples()">
        <button onclick="expandAll()">Expand All</button>
        <button onclick="collapseAll()">Collapse All</button>
    </div>
    <div id="samples">
'''

    for i, r in enumerate(results):
        sample_id = r.get("sample_id", f"sample-{i}")
        gts = r.get("gts", "")
        saved_images = r.get("saved_images", [])

        # Keys to skip (large or not useful for display)
        skip_keys = {"sample_id", "dataset", "input", "output", "gts", "saved_images"}

        # Display all keys equally in meta-row
        meta_items = []
        meta_items.append(f'<div class="meta-item"><span class="meta-label">Ground Truth:</span> <span class="meta-value">{html_lib.escape(str(gts))}</span></div>')

        for key, value in r.items():
            if key in skip_keys:
                continue
            # Format values
            if value is None:
                formatted = "None"
            elif isinstance(value, float):
                formatted = f"{value:.4f}"
            elif isinstance(value, (int, bool)):
                formatted = str(value)
            elif isinstance(value, str) and len(value) < 200:
                formatted = html_lib.escape(value)
            else:
                continue  # Skip complex or long values
            meta_items.append(f'<div class="meta-item"><span class="meta-label">{html_lib.escape(key)}:</span> <span class="meta-value">{formatted}</span></div>')

        # Process input and output with image replacements
        input_html = highlight_tags_with_images(r.get("input", ""), saved_images)
        output_html = highlight_tags_with_images(r.get("output", ""), saved_images)

        html_content += f'''
        <div class="sample collapsed" id="sample-{i}">
            <div class="sample-header" onclick="toggle({i})">
                <span class="toggle">&#9660;</span>
                <span class="sample-title">{html_lib.escape(sample_id)}</span>
            </div>
            <div class="sample-content">
                <div class="meta-row">{"".join(meta_items)}</div>
                <div class="conversation">
                    <div class="turn user">
                        <div class="turn-header">Input</div>
                        <div class="turn-content">{input_html}</div>
                    </div>
                    <div class="turn assistant">
                        <div class="turn-header">Output</div>
                        <div class="turn-content">{output_html}</div>
                    </div>
                </div>
            </div>
        </div>
'''

    html_content += '''
    </div>
    <script>
        function toggle(i) {
            document.getElementById('sample-' + i).classList.toggle('collapsed');
        }
        function expandAll() {
            document.querySelectorAll('.sample').forEach(s => s.classList.remove('collapsed'));
        }
        function collapseAll() {
            document.querySelectorAll('.sample').forEach(s => s.classList.add('collapsed'));
        }
        function filterSamples() {
            const query = document.getElementById('search').value.toLowerCase();
            document.querySelectorAll('.sample').forEach(s => {
                s.style.display = s.textContent.toLowerCase().includes(query) ? '' : 'none';
            });
        }
    </script>
</body>
</html>'''

    with open(output_path, "w") as f:
        f.write(html_content)


def save_results(dataset_results: dict, output_dir: str):
    """Save summary to output directory. Note: results.jsonl is saved incrementally during eval."""
    os.makedirs(output_dir, exist_ok=True)

    # Summary only (results.jsonl already saved incrementally)
    summary = {"datasets": dataset_results}

    # Add web fetch stats if any
    ws = _web_fetch_stats.get_stats()
    if ws["total"] > 0:
        summary["web_fetch_stats"] = ws

    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {output_dir}", flush=True)
    print(f"  Results: {output_dir}/results.jsonl", flush=True)
    html_path = os.path.join(output_dir, "results.html")
    if os.path.exists(html_path):
        print(f"  HTML:    {html_path}", flush=True)
    print(f"  Summary: {output_dir}/summary.json", flush=True)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="VLM Evaluation Script v2")

    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--mode", type=str, required=True, choices=["direct", "tool"])
    parser.add_argument("--datasets", type=str, required=True, help="Path to datasets JSON config")
    parser.add_argument("--model-client", type=str, choices=["gemini", "openai", "azure"], required=True,
                        help="Model API client: gemini, openai (Qwen/MARS), or azure (GPT)")
    parser.add_argument("--judge-client", type=str, choices=["openai", "azure"], default=None,
                        help="Judge API client: openai or azure (required if using LLM judge)")
    parser.add_argument("--judge-temperature", type=float, default=0.0,
                        help="Temperature for LLM judge (default: 0.0)")
    parser.add_argument("--data-root", type=str, default="", help="Root for relative paths")
    parser.add_argument("--output-dir", type=str, default=None)

    parser.add_argument("--max-concurrent", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-turns", type=int, default=50,
                        help="Max assistant turns (default: 50)")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--presence-penalty", type=float, default=1.5)
    parser.add_argument("--repetition-penalty", type=float, default=1.0,
                        help="Repetition penalty for model generation (default: 1.0)")
    parser.add_argument("--seed", type=int, default=3407,
                        help="Random seed for deterministic generation (default: 3407)")

    parser.add_argument("--min-pixels", type=int, default=65536, help="Min pixels for image processing")
    parser.add_argument("--max-pixels", type=int, default=8294400, help="Max pixels for image processing")
    parser.add_argument("--factor", type=int, default=32, help="Alignment factor (32 for Qwen3-VL, 28 for Qwen2-VL)")
    parser.add_argument("--qwen-vl-processing", type=lambda x: x.lower() == 'true', default=True,
                        help="Use Qwen-VL style image processing (default: True, set False for Gemini/GPT)")
    parser.add_argument("--tool-config", type=str, default="", help="Path to tool config YAML (optional, overrides dataset system_prompt)")
    parser.add_argument("--serper-concurrency", type=int, default=5, help="Max concurrent Serper API requests (default: 5)")
    parser.add_argument("--search-cache-dir", type=str, default="", help="Directory for search result cache (optional, no cache if not set)")

    args = parser.parse_args()

    model_client = args.model_client

    # Get API credentials based on model_client
    if model_client == "gemini":
        if "GEMINI_API_KEY" not in os.environ:
            parser.error("GEMINI_API_KEY required")
        if "GEMINI_BASE_URL" not in os.environ:
            parser.error("GEMINI_BASE_URL required")
        api_key = os.environ["GEMINI_API_KEY"]
        base_url = os.environ["GEMINI_BASE_URL"]
    elif model_client == "azure":
        if "AZURE_OPENAI_API_KEY" not in os.environ:
            parser.error("AZURE_OPENAI_API_KEY required")
        if "AZURE_OPENAI_BASE_URL" not in os.environ:
            parser.error("AZURE_OPENAI_BASE_URL required")
        api_key = os.environ["AZURE_OPENAI_API_KEY"]
        base_url = os.environ["AZURE_OPENAI_BASE_URL"]
    else:
        # OpenAI-compatible (Qwen, MARS, etc.)
        if "MODEL_BASE_URL" not in os.environ:
            parser.error("MODEL_BASE_URL required")
        api_key = ""
        base_url = os.environ["MODEL_BASE_URL"]

    # Load datasets
    print(f"Loading datasets from {args.datasets}...")
    samples, dataset_configs = load_datasets(args.datasets, args.data_root)
    print(f"Loaded {len(samples)} samples total\n")

    # Check LLM judge requirement
    needs_llm = any("llm_score" in cfg.get("score_methods", [])
                    for cfg in dataset_configs.values())
    judge_client = args.judge_client or ""
    judge_base_url = ""
    judge_api_key = ""
    if needs_llm:
        if not args.judge_client:
            parser.error("--judge-client required when using LLM judge (llm_score)")
        if judge_client == "azure":
            if "AZURE_OPENAI_API_KEY" not in os.environ:
                parser.error("AZURE_OPENAI_API_KEY required for judge")
            if "AZURE_OPENAI_BASE_URL" not in os.environ:
                parser.error("AZURE_OPENAI_BASE_URL required for judge")
            judge_base_url = os.environ["AZURE_OPENAI_BASE_URL"]
            judge_api_key = os.environ["AZURE_OPENAI_API_KEY"]
        else:  # openai (official API)
            if "OPENAI_API_KEY" not in os.environ:
                parser.error("OPENAI_API_KEY required for judge")
            judge_api_key = os.environ["OPENAI_API_KEY"]

    # Get serper key and summarizer config for tool mode
    serper_api_key = ""
    summarizer_base_url = ""
    summarizer_model = ""
    tools_section = ""
    if args.mode == "tool":
        if not args.tool_config:
            parser.error("--tool-config is required for tool mode")
        if "SERPER_API_KEY" not in os.environ:
            parser.error("SERPER_API_KEY required for tool mode")
        if "SUMMARIZER_BASE_URL" not in os.environ:
            parser.error("SUMMARIZER_BASE_URL required for tool mode")
        if "SUMMARIZER_MODEL" not in os.environ:
            parser.error("SUMMARIZER_MODEL required for tool mode")
        serper_api_key = os.environ["SERPER_API_KEY"]
        summarizer_base_url = os.environ["SUMMARIZER_BASE_URL"]
        summarizer_model = os.environ["SUMMARIZER_MODEL"]
        print(f"Loading tool config from {args.tool_config}...")
        tools_section = load_tool_config(args.tool_config)

    # Output dir
    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = time.strftime("%y%m%d%H%M%S")
        output_dir = f"eval_{args.model.replace('/', '_')}_{args.mode}_{timestamp}"

    # Create search cache if directory provided
    search_cache = None
    if args.search_cache_dir:
        search_cache = SearchCache(args.search_cache_dir)

    # Run evaluation
    kwargs = {
        "max_tokens": args.max_tokens,
        "max_turns": args.max_turns,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "presence_penalty": args.presence_penalty,
        "repetition_penalty": args.repetition_penalty,
        "seed": args.seed,
        "min_pixels": args.min_pixels,
        "max_pixels": args.max_pixels,
        "factor": args.factor,
        "qwen_vl_processing": args.qwen_vl_processing,
        "serper_api_key": serper_api_key,
        "summarizer_base_url": summarizer_base_url,
        "summarizer_model": summarizer_model,
        "serper_concurrency": args.serper_concurrency,
        "search_cache": search_cache,
        "tools_section": tools_section,
        "judge_client": judge_client,
        "judge_base_url": judge_base_url,
        "judge_api_key": judge_api_key,
        "judge_temperature": args.judge_temperature,
    }

    # Health check - crash early if servers are down
    async def health_check():
        import aiohttp
        print("Running health checks...", flush=True)

        # Check model server
        print(f"  Checking model server: {base_url}", flush=True)
        try:
            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                # Try /v1/models endpoint (OpenAI-compatible)
                url = f"{base_url.rstrip('/')}/v1/models"
                headers = {"Content-Type": "application/json"}
                if api_key:
                    headers["Authorization"] = f"Bearer {api_key}"
                async with session.get(url, headers=headers) as resp:
                    if resp.status != 200:
                        raise Exception(f"Model server returned HTTP {resp.status}")
                    print(f"    Model server OK", flush=True)
        except Exception as e:
            raise RuntimeError(f"Model server not reachable at {base_url}: {e}")

        # Check summarizer server (if in tool mode)
        if args.mode == "tool" and summarizer_base_url:
            print(f"  Checking summarizer server: {summarizer_base_url}", flush=True)
            try:
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    url = f"{summarizer_base_url.rstrip('/')}/v1/models"
                    async with session.get(url) as resp:
                        if resp.status != 200:
                            raise Exception(f"Summarizer server returned HTTP {resp.status}")
                        print(f"    Summarizer server OK", flush=True)
            except Exception as e:
                raise RuntimeError(f"Summarizer server not reachable at {summarizer_base_url}: {e}")

        print("Health checks passed!", flush=True)

    asyncio.run(health_check())

    try:
        dataset_results = asyncio.run(run_evaluation(
            samples, dataset_configs, model_client, args.model, base_url, api_key,
            args.mode, args.max_concurrent, output_dir=output_dir, **kwargs
        ))

        save_results(dataset_results, output_dir)
    finally:
        # Skip browser cleanup - Playwright can hang indefinitely and OS will clean up on exit
        # Always print cache stats and close, even on error
        if search_cache:
            stats = search_cache.get_stats()
            print(f"Search cache: {stats['hits']}/{stats['total']} hits ({stats['hit_rate']:.1f}%), {stats['misses']} new searches cached", flush=True)
            search_cache.close()
        print("Done.", flush=True)


if __name__ == "__main__":
    main()
