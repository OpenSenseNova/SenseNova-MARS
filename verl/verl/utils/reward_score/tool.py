# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import base64
import copy
import io
import re
import string
import random
import json
import threading

# =============================================================================
# Module-level counters for LLM judge API call tracking
# =============================================================================
_judge_llm_lock = threading.Lock()
_judge_llm_total = 0
_judge_llm_successful = 0

# =============================================================================
# Module-level semaphore for rate limiting LLM judge API calls
# This limits concurrent requests WITHIN each worker process.
# Total max concurrent = num_workers * semaphore_limit
# =============================================================================
_judge_llm_semaphore = None
_judge_llm_semaphore_limit = 0


def configure_judge_llm_semaphore(limit: int):
    """Configure the semaphore limit for LLM judge API calls.

    Call this once during initialization to set the concurrency limit.
    If limit <= 0, no rate limiting is applied.
    If already configured with same limit, this is a no-op.
    If already configured with different limit, logs a warning but keeps the original.

    Args:
        limit: Max concurrent LLM judge requests per worker process.
               Recommended: 1-4 depending on vLLM server capacity.
    """
    global _judge_llm_semaphore, _judge_llm_semaphore_limit

    # Already configured - check if same limit
    if _judge_llm_semaphore is not None:
        if _judge_llm_semaphore_limit == limit:
            return  # Same limit, no-op
        else:
            print(
                f"[JudgeLLM] WARNING: Semaphore already configured with limit={_judge_llm_semaphore_limit}, "
                f"ignoring new limit={limit}. Use the same limit for train and val.",
                flush=True
            )
            return

    if limit > 0:
        _judge_llm_semaphore = threading.Semaphore(limit)
        _judge_llm_semaphore_limit = limit
        print(f"[JudgeLLM] Semaphore configured: max {limit} concurrent requests per worker", flush=True)
    else:
        _judge_llm_semaphore = None
        _judge_llm_semaphore_limit = 0


def get_judge_llm_stats(reset: bool = False) -> dict:
    """Get current LLM judge call statistics.

    Args:
        reset: If True, reset counters after reading

    Returns:
        dict with keys: total, successful, success_rate (0.0-1.0)
    """
    global _judge_llm_total, _judge_llm_successful
    with _judge_llm_lock:
        total = _judge_llm_total
        successful = _judge_llm_successful
        success_rate = (successful / total) if total > 0 else 1.0
        if reset:
            _judge_llm_total = 0
            _judge_llm_successful = 0
    return {
        "judge_llm_total": total,
        "judge_llm_successful": successful,
        "judge_llm_success_rate": success_rate,
    }


def reset_judge_llm_stats():
    """Reset LLM judge call counters."""
    global _judge_llm_total, _judge_llm_successful
    with _judge_llm_lock:
        _judge_llm_total = 0
        _judge_llm_successful = 0


def check_judge_llm_success_rate(threshold: float = 0.95, context: str = ""):
    """Check if LLM judge success rate is above threshold, crash if not.

    Args:
        threshold: Minimum acceptable success rate (0.0-1.0, default 0.95)
        context: Additional context for error message (e.g., "train" or "val")

    Raises:
        RuntimeError: If success rate is below threshold
    """
    stats = get_judge_llm_stats(reset=False)  # Don't reset - let caller decide
    total = stats["judge_llm_total"]
    successful = stats["judge_llm_successful"]
    success_rate = stats["judge_llm_success_rate"]

    if total > 0:
        ctx_str = f"[{context}] " if context else ""
        print(
            f"[JudgeLLM] {ctx_str}Stats: total={total}, successful={successful}, "
            f"success_rate={success_rate:.2%}",
            flush=True
        )

        if success_rate < threshold:
            error_msg = (
                f"[JudgeLLM] {ctx_str}FATAL: Success rate {success_rate:.2%} is below "
                f"threshold {threshold:.0%} ({successful}/{total} calls succeeded). "
                f"This indicates LLM judge API issues (timeout/rate limit/server error). "
                f"Crashing to prevent training on incomplete reward data."
            )
            print(f"\n{'='*80}\n{error_msg}\n{'='*80}\n", flush=True)
            raise RuntimeError(error_msg)


def _increment_judge_llm_total():
    """Increment total LLM judge call counter (thread-safe)."""
    global _judge_llm_total
    with _judge_llm_lock:
        _judge_llm_total += 1


def _increment_judge_llm_successful():
    """Increment successful LLM judge call counter (thread-safe)."""
    global _judge_llm_successful
    with _judge_llm_lock:
        _judge_llm_successful += 1


def convert_hf_messages_to_openai(messages, images):
    """
    Convert HF format messages to OpenAI format with base64-encoded images.

    HF format: {"type": "image"} with separate PIL images list
    OpenAI format: {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}

    Args:
        messages: List of message dicts in HF format
        images: List of PIL Image objects

    Returns:
        List of message dicts in OpenAI format
    """
    if not images:
        return messages

    messages = copy.deepcopy(messages)
    image_idx = 0

    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue

        new_content = []
        for item in content:
            if item.get("type") == "image":
                # Convert PIL image to base64
                if image_idx < len(images):
                    img = images[image_idx]
                    image_idx += 1

                    # Encode to JPEG base64 (smaller file size, quality 85 is sufficient for LLM Judge)
                    buffer = io.BytesIO()
                    # Convert to RGB if necessary (JPEG doesn't support RGBA)
                    if img.mode in ('RGBA', 'LA', 'P'):
                        img = img.convert('RGB')
                    img.save(buffer, format="JPEG", quality=85)
                    img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

                    new_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}
                    })
                else:
                    # No more images available, skip this placeholder
                    pass
            elif item.get("type") == "image_url":
                # Already in OpenAI format, keep as-is
                new_content.append(item)
            else:
                new_content.append(item)

        message["content"] = new_content

    return messages


def get_user_message(prompts):
    """Get first user message from prompts list, works with or without system message."""
    for msg in prompts:
        if msg['role'] == 'user':
            return msg
    raise ValueError("No user message found in prompts")


def get_question_from_user_message(user_msg):
    """Extract question text from user message content.

    Handles both string content and list content (with text/image items).
    For list content, returns the first text item found.
    """
    content = user_msg.get('content', '')

    # Handle string content (text-only prompts)
    if isinstance(content, str):
        return content

    # Handle list content (multimodal prompts)
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and item.get('type') == 'text':
                return item.get('text', '')
            # Also handle simple string items in list
            if isinstance(item, str):
                return item

    raise ValueError(f"Could not extract question text from user message content: {type(content)}")


def normalize_answer(s):
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


def em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1
            break
    return score

def extract_solution(solution_str):
    """
    Simple extractor - extracts raw answer from <answer> tags only.
    Returns None if no <answer> tag found.

    Note: For MCQ datasets where you need to extract choice letters (A-D),
    use extract_solution_for_mcq() instead.
    """
    # Aligned with Format V5 validation regex: (.*?) allows < chars (e.g., math "5 < 10")
    matches = list(re.finditer(r'<answer>(.*?)</answer>', solution_str, re.DOTALL))
    if matches:
        return matches[-1].group(1).strip()
    return None


def extract_solution_allow_no_answer_tag(solution_str):
    """
    Extractor that allows no answer tag - falls back to entire response.

    If <answer>...</answer> tags are found, returns the last match.
    Otherwise, returns the entire response (up to <|im_end|> if present).

    This is useful for evaluating models that may not always produce
    structured answer tags but still provide valid responses.
    """
    assert "<|im_start|>assistant" not in solution_str

    # First try to extract from <answer> tags
    matches = list(re.finditer(r'<answer>(.*?)</answer>', solution_str, re.DOTALL))

    # If there are matches, return the last one
    if len(matches) >= 1:
        return matches[-1].group(1).strip()

    # Fallback: return entire response (stripped at <|im_end|> if present)
    return solution_str.split('<|im_end|>')[0].strip()


def compute_score_em(solution_str, ground_truth, method='strict', format_score=0., score=1., **kwargs):
    """The scoring function for exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    # Handle None or empty solution_str
    if solution_str is None:
        solution_str = ""
    solution_str = solution_str.rsplit("<|im_start|>assistant", 1)[-1]
    solution_str = re.split(r'</think(?:ing)?>', solution_str)[-1]  # Support both </think> and </thinking>
    answer = extract_solution(solution_str=solution_str)  # 正则表达式提取答案 <answer>(.*?)</answer>
    do_print = random.randint(1, 64) == 1

    if answer is None:
        score = 0
        em_match = False
    else:
        if em_check(answer, ground_truth):
            score = 1
            em_match = True
        else:
            score = 0
            em_match = False

    if do_print:
        log_msg = "\n".join([
            "",
            "=" * 80,
            "EM SCORE",
            "=" * 80,
            f"Answer extracted: {answer is not None}",
            f"EM match: {em_match}",
            f"Golden answers: {ground_truth}",
            f"Extracted answer: {answer}",
            f"Solution string: {solution_str}",
            f"Score: {score}",
            "=" * 80,
            ""
        ])
        print(log_msg, flush=True)

    return score


def compute_score_f1(solution_str, ground_truth, **kwargs):
    """
    Compute F1 score between predicted and ground truth answers.
    """
    # Handle None or empty solution_str
    if solution_str is None:
        solution_str = ""
    solution_str = solution_str.rsplit("<|im_start|>assistant", 1)[-1]
    solution_str = re.split(r'</think(?:ing)?>', solution_str)[-1]  # Support both </think> and </thinking>
    answer = extract_solution(solution_str=solution_str)  # 正则表达式提取答案 <answer>(.*?)</answer>

    if answer is None:
        f1_score = 0
    else:
        prediction_tokens = normalize_answer(answer).split()
        f1_score = 0
        if isinstance(ground_truth, str):
            ground_truth = [ground_truth]

        for cur_ground_truth in ground_truth:
            ground_truth_tokens = normalize_answer(cur_ground_truth).split()

            common = set(prediction_tokens) & set(ground_truth_tokens)
            num_same = len(common)

            if num_same == 0:
                cur_f1_score = 0
            else:
                precision = num_same / len(prediction_tokens)
                recall = num_same / len(ground_truth_tokens)
                cur_f1_score = 2 * precision * recall / (precision + recall)
            f1_score = max(f1_score, cur_f1_score)

    do_print = random.randint(1, 64) == 1
    if do_print:
        log_msg = "\n".join([
            "",
            "=" * 80,
            "F1 SCORE",
            "=" * 80,
            f"Answer extracted: {answer is not None}",
            f"Golden answers: {ground_truth}",
            f"Extracted answer: {answer}",
            f"Solution string: {solution_str}",
            f"Score: {f1_score}",
            "=" * 80,
            ""
        ])
        print(log_msg, flush=True)

    return f1_score


def compute_score_llm(solution_str, ground_truth, raw_prompt, clients, llm_config, multi_modal_data=None, **kwargs):
    """LLM-as-judge scoring function.

    Args:
        clients: List of OpenAI-compatible client objects
        llm_config: Dict with keys: model, temperature, max_tokens, timeout
    """
    # Single debug flag for all debug output (INPUT and OUTPUT print together)
    do_print = random.randint(1, 64) == 1

    # Validate configuration
    if not clients:
        raise ValueError("compute_score_llm requires clients but got empty list")
    if not llm_config:
        raise ValueError("compute_score_llm requires llm_config but got None")

    # Handle None or empty solution_str
    if solution_str is None:
        solution_str = ""
    solution_str = solution_str.rsplit("<|im_start|>assistant", 1)[-1]
    solution_str = re.split(r'</think(?:ing)?>', solution_str)[-1]  # Support both </think> and </thinking>
    answer = extract_solution(solution_str=solution_str)  # 正则表达式提取答案 <answer>(.*?)</answer>

    candidate_answers_str = ""  # Reserved for future use with candidate answers
    model_response = answer
    user_msg = get_user_message(raw_prompt)
    question = get_question_from_user_message(user_msg)

    # Extract LLM configuration (no defaults - fail if missing)
    llm_model = llm_config['model']
    llm_temperature = llm_config['temperature']
    llm_max_tokens = llm_config['max_tokens']
    llm_timeout = llm_config['timeout']
    is_vision_model = llm_config.get('is_vision_model', True)  # Default True for backward compatibility
    enable_thinking = llm_config.get('enable_thinking', False)  # Default False to disable thinking tokens

    # Different system messages for vision vs text-only models
    if is_vision_model:
        system_message = """You are an AI assistant tasked with evaluating the correctness of model responses based on an image, question, and ground truth answer. Your judgment should follow these principles:

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
    else:
        # Text-only model - no mention of images
        system_message = """You are an AI assistant tasked with evaluating the correctness of model responses based on a question and ground truth answer. Your judgment should follow these principles:

1. Consider the question and ground truth answer holistically before evaluating the model's response.
2. Your decision should be strictly Yes or No, based on whether the model's response is factually accurate and aligns with the ground truth answer.
3. If the model response is a more specific form of the ground truth answer, it is correct.
4. If the model response includes all key information but adds minor details, it is correct as long as the extra details are factually correct.
5. If the model response contradicts, modifies, or omits critical parts of the answer, it is incorrect.
6. For numerical values, ensure correctness even when presented in different units.
7. For names, check for first and last name correctness. If the middle name is extra but correct, consider it correct.
8. For yes/no questions, the response must exactly match "Yes" or "No" to be correct.
9. If there are multiple candidate answers, you can also evaluate the model's response against all of them. If the response aligns with at least one candidate according to the rules above, it should be considered correct.
10. For multiple choice questions (A, B, C, D), be more lenient. If the model provides the correct letter choice, even with additional text or formatting, consider it correct.
11. If the model's answer contains the correct choice letter (A, B, C, or D) anywhere in the response, and it's clear this is the intended answer, mark it as correct.
12. Ignore formatting issues like extra parentheses, brackets, or minor text variations as long as the core answer is correct.

Your output must be in the following format:
<judge>Yes/No</judge>
<reason>Explanation of why the answer is correct or incorrect.</reason>"""

    # Different prompts for vision vs text-only models
    # Adjust prompt based on whether candidate answers are provided
    has_candidate_answers = bool(candidate_answers_str.strip())

    if is_vision_model:
        if has_candidate_answers:
            prompt = """Image, Question, and Model Response Evaluation
Question: {question}
Ground Truth Answer: {ground_truth_answer}
{candidate_answers_str}Model Response: {model_response}

Evaluation Instructions
Evaluate whether the Model Response is correct based on the Image, Question, Ground Truth Answer and Candidate Answers. Follow the predefined judgment rules and provide a clear Yes/No answer along with a justification.

Output Format
<judge>Yes/No</judge>
<reason>Detailed reasoning following the evaluation principles.</reason>"""
        else:
            prompt = """Image, Question, and Model Response Evaluation
Question: {question}
Ground Truth Answer: {ground_truth_answer}
Model Response: {model_response}

Evaluation Instructions
Evaluate whether the Model Response is correct based on the Image, Question and Ground Truth Answer. Follow the predefined judgment rules and provide a clear Yes/No answer along with a justification.

Output Format
<judge>Yes/No</judge>
<reason>Detailed reasoning following the evaluation principles.</reason>"""
    else:
        if has_candidate_answers:
            prompt = """Question and Model Response Evaluation
Question: {question}
Ground Truth Answer: {ground_truth_answer}
{candidate_answers_str}Model Response: {model_response}

Evaluation Instructions
Evaluate whether the Model Response is correct based on the Question, Ground Truth Answer and Candidate Answers. Follow the predefined judgment rules and provide a clear Yes/No answer along with a justification.

Output Format
<judge>Yes/No</judge>
<reason>Detailed reasoning following the evaluation principles.</reason>"""
        else:
            prompt = """Question and Model Response Evaluation
Question: {question}
Ground Truth Answer: {ground_truth_answer}
Model Response: {model_response}

Evaluation Instructions
Evaluate whether the Model Response is correct based on the Question and Ground Truth Answer. Follow the predefined judgment rules and provide a clear Yes/No answer along with a justification.

Output Format
<judge>Yes/No</judge>
<reason>Detailed reasoning following the evaluation principles.</reason>"""

    evaluation_text = None  # Initialize for logging
    scoring_method = None  # Track: "no_answer", "em_match", or "llm_judge"

    if answer is None:
        llm_score = 0
        scoring_method = "no_answer"
    else:
        if em_check(answer, ground_truth):
            llm_score = 1.0
            scoring_method = "em_match"
        else:
            scoring_method = "llm_judge"
            # 只有EM不匹配时才使用LLM评判
            # prediction_tokens = normalize_answer(answer).split()

            llm_score = 0.0
            
            if isinstance(ground_truth, str):
                ground_truth = [ground_truth]

            for gt in ground_truth:
                # Format prompt with or without candidate_answers_str
                if has_candidate_answers:
                    filled_prompt = prompt.format(question=question, ground_truth_answer=gt, candidate_answers_str=candidate_answers_str, model_response=model_response)
                else:
                    filled_prompt = prompt.format(question=question, ground_truth_answer=gt, model_response=model_response)

                # Construct messages - conditionally include image based on model type
                if is_vision_model:
                    # Vision model: include image
                    # Convert HF format to OpenAI format for API compatibility
                    images = multi_modal_data.get("image", []) if multi_modal_data else []
                    converted_prompt = convert_hf_messages_to_openai(raw_prompt, images)
                    user_msg = get_user_message(converted_prompt)
                    # Get image content (now in OpenAI format: {"type": "image_url", ...})
                    image_content = None
                    for item in user_msg.get('content', []):
                        if item.get('type') == 'image_url':
                            image_content = item
                            break
                    if image_content:
                        messages = [
                            {"role": "system", "content": system_message},
                            {"role": "user", "content": [{'type': 'text', 'text': filled_prompt}, image_content]}
                        ]
                    else:
                        # No image found, fall back to text-only
                        messages = [
                            {"role": "system", "content": system_message},
                            {"role": "user", "content": filled_prompt}
                        ]
                else:
                    # Text-only model: text only
                    messages = [
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": filled_prompt}
                    ]

                # Random client selection
                client = clients[random.randrange(len(clients)) if len(clients) > 1 else 0]

                # Prepare API call parameters
                # chat_template_kwargs is vLLM-specific and only supported by Qwen models
                api_params = {
                    "model": llm_model,
                    "messages": messages,
                    "max_tokens": llm_max_tokens,
                    "temperature": llm_temperature,
                    "timeout": llm_timeout,
                }

                # Only add extra_body for Qwen3 models (vLLM backend)
                # Azure OpenAI and other providers don't support chat_template_kwargs
                if 'qwen3' in llm_model.lower():
                    api_params["extra_body"] = {"chat_template_kwargs": {"enable_thinking": enable_thinking}}

                # Track LLM judge API calls for success rate monitoring
                _increment_judge_llm_total()

                # Log judge model and endpoint (1 in 64 calls)
                if do_print:
                    print(f"[LLM Judge] model={llm_model}, endpoint={client.base_url}", flush=True)

                # Use semaphore to limit concurrent requests (if configured)
                if _judge_llm_semaphore is not None:
                    _judge_llm_semaphore.acquire()
                try:
                    response = client.chat.completions.create(**api_params)
                    # API call succeeded - increment successful counter
                    _increment_judge_llm_successful()
                except Exception as e:
                    # API call failed - don't increment successful counter
                    # Convert to Ray-serializable exception to preserve error details
                    # OpenAI's APIStatusError has required keyword-only args that can't be pickled
                    # After 5 retries failed, this is a serious issue that should fail loud
                    error_msg = f"LLM judge API call failed after 5 retries: {type(e).__name__}: {str(e)}"
                    print(f"[ERROR] {error_msg}", flush=True)
                    raise RuntimeError(error_msg) from None
                finally:
                    if _judge_llm_semaphore is not None:
                        _judge_llm_semaphore.release()

                # Parse response
                evaluation_text = response.choices[0].message.content.strip()

                # 在整个返回文本中搜索judge标签
                judge_match = re.search(r'<judge>\s*(Yes|No)\s*</judge>', evaluation_text, re.IGNORECASE | re.DOTALL)

                judge_decision = "Yes" if judge_match and judge_match.group(1).lower() == 'yes' else "No"
                if judge_decision == "Yes":
                    llm_score = max(llm_score, 1.0)
                else:
                    # 如果找不到标签或者标签内容是'No'，则得分为0
                    llm_score = max(llm_score, 0.0)

                # Print INPUT + OUTPUT together (after API call returns, so they're not interleaved)
                if do_print:
                    # Build messages content string for atomic print
                    msg_lines = []
                    img_counter = 0
                    for msg in messages:
                        role = msg.get("role", "unknown")
                        content = msg.get("content", "")
                        msg_lines.append(f"--- {role} ---")
                        if isinstance(content, list):
                            # Multi-modal content - reconstruct with image placeholders
                            for part in content:
                                if part.get("type") == "text":
                                    msg_lines.append(part.get("text", ""))
                                elif part.get("type") == "image_url":
                                    img_counter += 1
                                    msg_lines.append(f"[IMAGE {img_counter}]")
                        else:
                            msg_lines.append(content)
                    messages_str = "\n".join(msg_lines)

                    log_msg = "\n".join([
                        "",
                        "=" * 80,
                        "LLM SCORE",
                        "=" * 80,
                        "Method: LLM-as-a-Judge",
                        "Answer extracted: True",
                        "EM match: False",
                        f"Golden answers: {ground_truth}",
                        f"Extracted answer: {answer}",
                        f"Solution string: {solution_str}",
                        "-" * 40,
                        "[INPUT] Messages sent to vLLM:",
                        messages_str,
                        "-" * 40,
                        "[OUTPUT] vLLM response:",
                        evaluation_text,
                        "-" * 40,
                        f"Judge decision: {judge_decision}",
                        f"Score: {llm_score}",
                        "=" * 80,
                        ""
                    ])
                    print(log_msg, flush=True)

            # evaluation_text = response.choices[0].message.content.strip()
            # lines = evaluation_text.split('\n')
            # decision = lines[0].strip()

            # # Extract Yes/No from decision (handle various formats)
            # if decision.lower().startswith('<judge>'):
            #     judge_match = re.search(r'<judge>(.*?)</judge>', decision, re.IGNORECASE)
            #     if judge_match:
            #         decision = judge_match.group(1).strip()
 
            # if decision.lower() in ['yes', '<judge>yes</judge>']:
            #     llm_score = max(llm_score, 1.0)
            # else:
            #     llm_score = max(llm_score, 0.0)

            # do_print = random.randint(1, 64) == 1
            # if do_print:
            #     print(f"filled_prompt: {filled_prompt}")
            #     print(f"judge response: {response}")
            #     print(f"llm score: {llm_score}")
    
    # Use same do_print flag set at function start (always defined)
    # Print summary for non-LLM paths (no_answer and em_match)
    # LLM path already printed INPUT+OUTPUT together inside the loop
    if do_print and scoring_method != "llm_judge":
        log_msg = "\n".join([
            "",
            "=" * 80,
            "LLM SCORE",
            "=" * 80,
            "Method: EM",
            f"Answer extracted: {answer is not None}",
            f"EM match: {scoring_method == 'em_match'}",
            f"Golden answers: {ground_truth}",
            f"Extracted answer: {answer}",
            f"Solution string: {solution_str}",
            f"Score: {llm_score}",
            "=" * 80,
            ""
        ])
        print(log_msg, flush=True)

    return llm_score


# =============================================================================
# Format Score V6: Constants and Helpers
# =============================================================================

# Error types that are the model's fault (should be penalized)
# - invalid_json: Model generated malformed JSON in tool call arguments
# - unknown_tool: Model called a tool that doesn't exist
# NOT included: execution_error (infrastructure/code bug, not model's fault)
_MODEL_CAUSED_ERROR_TYPES = frozenset({"invalid_json", "unknown_tool"})


def _has_model_caused_tool_error(data_non_tensor_batch: dict) -> bool:
    """
    Check if any tool call had a model-caused error.

    This checks two sources:
    1. tool_rewards: Negative reward indicates tool received invalid arguments
       (e.g., out-of-range coordinates, invalid bbox, image not found)
    2. tool_stats: error_type indicates agent-level errors
       (e.g., invalid JSON syntax, unknown tool name)

    Infrastructure errors (execution_error) are NOT considered model-caused.

    Args:
        data_non_tensor_batch: Dict containing tool_rewards and tool_stats

    Returns:
        True if any model-caused error was found, False otherwise
    """
    # Check 1: Tool argument errors (tools return negative reward for bad args)
    tool_rewards = data_non_tensor_batch.get("tool_rewards")
    if tool_rewards and any(r < 0 for r in tool_rewards):
        return True

    # Check 2: Agent-level model errors (invalid JSON, unknown tool)
    tool_stats = data_non_tensor_batch.get("tool_stats")
    if tool_stats and any(s.get("error_type") in _MODEL_CAUSED_ERROR_TYPES for s in tool_stats):
        return True

    return False


# =============================================================================
# Format Score Function
# =============================================================================

def compute_format_score(solution_str: str, ground_truth, format_score, is_valid_format=True, **kwargs) -> float:
    """
    Compute format score based on response validity and tool call success.

    Returns format_score if all conditions are met, otherwise returns 0:
    - is_valid_format is True (has answer tag, non-empty answer, valid JSON syntax)
    - No model-caused tool errors occurred

    Model-caused errors that result in 0 score:
    - Tool argument errors: out-of-range coordinates, invalid bbox, missing image
      (detected via negative tool_reward)
    - Agent-level errors: malformed JSON arguments, unknown tool name
      (detected via error_type in tool_stats)

    Infrastructure errors (execution_error) are NOT penalized.

    Args:
        solution_str: The model's response (unused, kept for API compatibility)
        ground_truth: The expected answer (unused, kept for API compatibility)
        format_score: The score to return if format is valid (typically 0.5)
        is_valid_format: Pre-computed format validation result
        **kwargs: Must contain 'data_non_tensor_batch' with tool_rewards and tool_stats

    Returns:
        format_score if valid, 0.0 otherwise
    """
    if not is_valid_format:
        return 0.0

    data_non_tensor_batch = kwargs.get("data_non_tensor_batch", {})
    if _has_model_caused_tool_error(data_non_tensor_batch):
        return 0.0

    return format_score


def extract_solution_for_mcq(solution_str):
    """
    Extract MCQ answer letter (A-D) from solution string.

    Uses a two-stage extraction process:

    Stage 1 - Extract candidate text (priority order):
        1) Content from last <answer>...</answer> tag
        2) Content from last \\boxed{...} - only if it contains a letter A-D
        3) Entire solution string (fallback)

    Stage 2 - Extract letter from candidate:
        For tag/boxed content:
            1) Single letter A-D
            2) Letter with punctuation: (A), [A], A., A), A] - last match
            3) Last standalone A-D at word boundary

        For entire string fallback:
            1) "Answer: (A)" pattern - last match (most explicit)
            2) "correct answer is (A)" or "answer is (A)" - last match
            3) Bold **...(A)...** - last match
            4) Last (A) in parentheses
            5) Last A) or A] format
            6) Last standalone A-D at word boundary

    Uses last match to correctly handle outputs where the model discusses
    options before stating its final answer.

    Returns:
        Uppercase letter A-D if found, None otherwise
    """
    # Stage 1: Extract candidate text
    candidate = None
    from_tag = False

    # 1. Try <answer> tags first
    matches = list(re.finditer(r'<answer>(.*?)</answer>', solution_str, re.DOTALL))
    if matches:
        candidate = matches[-1].group(1).strip()
        from_tag = True
    else:
        # 2. Try \boxed{} - BUT only if it contains a letter A-D
        match = re.search(r'\\boxed\{([^}]+)\}', solution_str, re.IGNORECASE)
        if match:
            boxed_content = match.group(1).strip()
            # Only treat as tag if boxed content has an MCQ letter
            if re.search(r'[A-Da-d]', boxed_content):
                candidate = boxed_content
                from_tag = True
            # else: fall through to entire string fallback

        if not from_tag:
            # 3. Fallback: use entire string
            candidate = solution_str
            from_tag = False

    # Stage 2: Extract letter from candidate
    stripped = candidate.strip()

    # For tag/boxed content, use direct extraction
    if from_tag:
        # 1. Single letter A-D
        if re.match(r'^[A-Da-d]$', stripped):
            return stripped.upper()

        # 2. Letter with punctuation: (A), [A], A., A), A] - LAST match
        punct_matches = re.findall(r'(?:\(([A-D])\)|\[([A-D])\]|(?<![A-Za-z])([A-D])[.\)\]])', stripped, re.IGNORECASE)
        if punct_matches:
            last = punct_matches[-1]
            return (last[0] or last[1] or last[2]).upper()

        # 3. Last standalone A-D at word boundary
        standalone_matches = re.findall(r'(?<![A-Za-z])([A-D])(?![A-Za-z])', stripped)
        if standalone_matches:
            return standalone_matches[-1].upper()

        return None

    # For entire string fallback (uses LAST match for all patterns)

    # 1. "Answer: (A)" pattern - most explicit, LAST match
    answer_matches = re.findall(r'Answer:\s*\(([A-D])\)', candidate, re.IGNORECASE)
    if answer_matches:
        return answer_matches[-1].upper()

    # 2. Phrase "correct answer is (A)" or "answer is (A)" - LAST match
    phrase_matches = re.findall(r'(?:correct answer is|answer is)[:\s]*\(([A-D])\)', candidate, re.IGNORECASE)
    if phrase_matches:
        return phrase_matches[-1].upper()

    # 3. Bold **...(A)...** - LAST match
    all_bold_segments = re.findall(r'\*\*[^*]+\*\*', candidate)
    bold_letters = []
    for seg in all_bold_segments:
        letter_match = re.search(r'\(([A-D])\)', seg, re.IGNORECASE)
        if letter_match:
            bold_letters.append(letter_match.group(1).upper())
    if bold_letters:
        return bold_letters[-1]

    # 4. Last (A) in parentheses
    paren_matches = re.findall(r'\(([A-D])\)', candidate, re.IGNORECASE)
    if paren_matches:
        return paren_matches[-1].upper()

    # 5. Last A) or A] format (NOT A. to avoid matching "data.")
    bracket_matches = re.findall(r'(?<![A-Za-z])([A-D])[\)\]]', candidate, re.IGNORECASE)
    if bracket_matches:
        return bracket_matches[-1].upper()

    # 6. Last standalone A-D at word boundary
    standalone_matches = re.findall(r'(?<![A-Za-z])([A-D])(?![A-Za-z])', candidate)
    if standalone_matches:
        return standalone_matches[-1].upper()

    return None


def compute_score_em_for_mcq(solution_str, ground_truth, method='strict', format_score=0., score=1., **kwargs):
    """
    Exact Match scoring for multiple choice questions.

    Extracts the model's answer choice (A-D) and compares it to the ground truth.
    Uses extract_solution_for_mcq which handles various answer formats including
    <answer> tags, \\boxed{}, bold text, and explicit answer statements.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth answer (letter A-D)
        method: extraction method (unused, kept for API compatibility)
        format_score: the score for the format (unused)
        score: the score for correct answer (unused)
    """
    solution_str = solution_str.rsplit("<|im_start|>assistant", 1)[-1]
    solution_str = re.split(r'</think(?:ing)?>', solution_str)[-1]
    answer = extract_solution_for_mcq(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1

    if answer is None:
        score = 0
        em_match = False
    else:
        if em_check(answer, ground_truth):
            score = 1
            em_match = True
        else:
            score = 0
            em_match = False

    if do_print:
        log_msg = "\n".join([
            "",
            "=" * 80,
            "EM SCORE (MCQ)",
            "=" * 80,
            f"Answer extracted: {answer is not None}",
            f"EM match: {em_match}",
            f"Golden answers: {ground_truth}",
            f"Extracted answer: {answer}",
            f"Solution string: {solution_str}",
            f"Score: {score}",
            "=" * 80,
            ""
        ])
        print(log_msg, flush=True)

    return score


def compute_score_llm_allow_no_answer_tag(solution_str, ground_truth, raw_prompt, clients, llm_config, multi_modal_data=None, **kwargs):
    """LLM-as-judge scoring function that allows no answer tag.

    This is similar to compute_score_llm but uses extract_solution_allow_no_answer_tag
    which falls back to the entire response if no <answer> tags are found.

    Args:
        clients: List of OpenAI-compatible client objects
        llm_config: Dict with keys: model, temperature, max_tokens, timeout
    """
    # Single debug flag for all debug output (INPUT and OUTPUT print together)
    do_print = random.randint(1, 64) == 1

    # Validate configuration
    if not clients:
        raise ValueError("compute_score_llm_allow_no_answer_tag requires clients but got empty list")
    if not llm_config:
        raise ValueError("compute_score_llm_allow_no_answer_tag requires llm_config but got None")

    # Handle None or empty solution_str
    if solution_str is None:
        solution_str = ""
    solution_str = solution_str.rsplit("<|im_start|>assistant", 1)[-1]
    solution_str = re.split(r'</think(?:ing)?>', solution_str)[-1]  # Support both </think> and </thinking>
    answer = extract_solution_allow_no_answer_tag(solution_str=solution_str)  # Uses fallback extraction

    candidate_answers_str = ""  # Reserved for future use with candidate answers
    model_response = answer
    user_msg = get_user_message(raw_prompt)
    question = get_question_from_user_message(user_msg)

    # Extract LLM configuration (no defaults - fail if missing)
    llm_model = llm_config['model']
    llm_temperature = llm_config['temperature']
    llm_max_tokens = llm_config['max_tokens']
    llm_timeout = llm_config['timeout']
    is_vision_model = llm_config.get('is_vision_model', True)  # Default True for backward compatibility
    enable_thinking = llm_config.get('enable_thinking', False)  # Default False to disable thinking tokens

    # Different system messages for vision vs text-only models
    if is_vision_model:
        system_message = """You are an AI assistant tasked with evaluating the correctness of model responses based on an image, question, and ground truth answer. Your judgment should follow these principles:

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
    else:
        # Text-only model - no mention of images
        system_message = """You are an AI assistant tasked with evaluating the correctness of model responses based on a question and ground truth answer. Your judgment should follow these principles:

1. Consider the question and ground truth answer holistically before evaluating the model's response.
2. Your decision should be strictly Yes or No, based on whether the model's response is factually accurate and aligns with the ground truth answer.
3. If the model response is a more specific form of the ground truth answer, it is correct.
4. If the model response includes all key information but adds minor details, it is correct as long as the extra details are factually correct.
5. If the model response contradicts, modifies, or omits critical parts of the answer, it is incorrect.
6. For numerical values, ensure correctness even when presented in different units.
7. For names, check for first and last name correctness. If the middle name is extra but correct, consider it correct.
8. For yes/no questions, the response must exactly match "Yes" or "No" to be correct.
9. If there are multiple candidate answers, you can also evaluate the model's response against all of them. If the response aligns with at least one candidate according to the rules above, it should be considered correct.
10. For multiple choice questions (A, B, C, D), be more lenient. If the model provides the correct letter choice, even with additional text or formatting, consider it correct.
11. If the model's answer contains the correct choice letter (A, B, C, or D) anywhere in the response, and it's clear this is the intended answer, mark it as correct.
12. Ignore formatting issues like extra parentheses, brackets, or minor text variations as long as the core answer is correct.

Your output must be in the following format:
<judge>Yes/No</judge>
<reason>Explanation of why the answer is correct or incorrect.</reason>"""

    # Different prompts for vision vs text-only models
    # Adjust prompt based on whether candidate answers are provided
    has_candidate_answers = bool(candidate_answers_str.strip())

    if is_vision_model:
        if has_candidate_answers:
            prompt = """Image, Question, and Model Response Evaluation
Question: {question}
Ground Truth Answer: {ground_truth_answer}
{candidate_answers_str}Model Response: {model_response}

Evaluation Instructions
Evaluate whether the Model Response is correct based on the Image, Question, Ground Truth Answer and Candidate Answers. Follow the predefined judgment rules and provide a clear Yes/No answer along with a justification.

Output Format
<judge>Yes/No</judge>
<reason>Detailed reasoning following the evaluation principles.</reason>"""
        else:
            prompt = """Image, Question, and Model Response Evaluation
Question: {question}
Ground Truth Answer: {ground_truth_answer}
Model Response: {model_response}

Evaluation Instructions
Evaluate whether the Model Response is correct based on the Image, Question and Ground Truth Answer. Follow the predefined judgment rules and provide a clear Yes/No answer along with a justification.

Output Format
<judge>Yes/No</judge>
<reason>Detailed reasoning following the evaluation principles.</reason>"""
    else:
        if has_candidate_answers:
            prompt = """Question and Model Response Evaluation
Question: {question}
Ground Truth Answer: {ground_truth_answer}
{candidate_answers_str}Model Response: {model_response}

Evaluation Instructions
Evaluate whether the Model Response is correct based on the Question, Ground Truth Answer and Candidate Answers. Follow the predefined judgment rules and provide a clear Yes/No answer along with a justification.

Output Format
<judge>Yes/No</judge>
<reason>Detailed reasoning following the evaluation principles.</reason>"""
        else:
            prompt = """Question and Model Response Evaluation
Question: {question}
Ground Truth Answer: {ground_truth_answer}
Model Response: {model_response}

Evaluation Instructions
Evaluate whether the Model Response is correct based on the Question and Ground Truth Answer. Follow the predefined judgment rules and provide a clear Yes/No answer along with a justification.

Output Format
<judge>Yes/No</judge>
<reason>Detailed reasoning following the evaluation principles.</reason>"""

    evaluation_text = None  # Initialize for logging
    scoring_method = None  # Track: "em_match" or "llm_judge"

    # Note: No "no_answer" case since extract_solution_allow_no_answer_tag always returns something
    if em_check(answer, ground_truth):
        llm_score = 1.0
        scoring_method = "em_match"
    else:
        scoring_method = "llm_judge"
        # 只有EM不匹配时才使用LLM评判

        llm_score = 0.0

        if isinstance(ground_truth, str):
            ground_truth = [ground_truth]

        for gt in ground_truth:
            # Format prompt with or without candidate_answers_str
            if has_candidate_answers:
                filled_prompt = prompt.format(question=question, ground_truth_answer=gt, candidate_answers_str=candidate_answers_str, model_response=model_response)
            else:
                filled_prompt = prompt.format(question=question, ground_truth_answer=gt, model_response=model_response)

            # Construct messages - conditionally include image based on model type
            if is_vision_model:
                # Vision model: include image
                # Convert HF format to OpenAI format for API compatibility
                images = multi_modal_data.get("image", []) if multi_modal_data else []
                converted_prompt = convert_hf_messages_to_openai(raw_prompt, images)
                user_msg = get_user_message(converted_prompt)
                # Get image content (now in OpenAI format: {"type": "image_url", ...})
                image_content = None
                for item in user_msg.get('content', []):
                    if item.get('type') == 'image_url':
                        image_content = item
                        break
                if image_content:
                    messages = [
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": [{'type': 'text', 'text': filled_prompt}, image_content]}
                    ]
                else:
                    # No image found, fall back to text-only
                    messages = [
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": filled_prompt}
                    ]
            else:
                # Text-only model: text only
                messages = [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": filled_prompt}
                ]

            # Random client selection
            client = clients[random.randrange(len(clients)) if len(clients) > 1 else 0]

            # Prepare API call parameters
            # chat_template_kwargs is vLLM-specific and only supported by Qwen models
            api_params = {
                "model": llm_model,
                "messages": messages,
                "max_tokens": llm_max_tokens,
                "temperature": llm_temperature,
                "timeout": llm_timeout,
            }

            # Only add extra_body for Qwen3 models (vLLM backend)
            # Azure OpenAI and other providers don't support chat_template_kwargs
            if 'qwen3' in llm_model.lower():
                api_params["extra_body"] = {"chat_template_kwargs": {"enable_thinking": enable_thinking}}

            # Track LLM judge API calls for success rate monitoring
            _increment_judge_llm_total()

            # Use semaphore to limit concurrent requests (if configured)
            if _judge_llm_semaphore is not None:
                _judge_llm_semaphore.acquire()
            try:
                response = client.chat.completions.create(**api_params)
                # API call succeeded - increment successful counter
                _increment_judge_llm_successful()
            except Exception as e:
                # API call failed - don't increment successful counter
                # Convert to Ray-serializable exception to preserve error details
                # OpenAI's APIStatusError has required keyword-only args that can't be pickled
                # After 5 retries failed, this is a serious issue that should fail loud
                error_msg = f"LLM judge API call failed after 5 retries: {type(e).__name__}: {str(e)}"
                print(f"[ERROR] {error_msg}", flush=True)
                raise RuntimeError(error_msg) from None
            finally:
                if _judge_llm_semaphore is not None:
                    _judge_llm_semaphore.release()

            # Parse response
            evaluation_text = response.choices[0].message.content.strip()

            # 在整个返回文本中搜索judge标签
            judge_match = re.search(r'<judge>\s*(Yes|No)\s*</judge>', evaluation_text, re.IGNORECASE | re.DOTALL)

            judge_decision = "Yes" if judge_match and judge_match.group(1).lower() == 'yes' else "No"
            if judge_decision == "Yes":
                llm_score = max(llm_score, 1.0)
            else:
                # 如果找不到标签或者标签内容是'No'，则得分为0
                llm_score = max(llm_score, 0.0)

            # Print INPUT + OUTPUT together (after API call returns, so they're not interleaved)
            if do_print:
                # Build messages content string for atomic print
                msg_lines = []
                img_counter = 0
                for msg in messages:
                    role = msg.get("role", "unknown")
                    content = msg.get("content", "")
                    msg_lines.append(f"--- {role} ---")
                    if isinstance(content, list):
                        # Multi-modal content - reconstruct with image placeholders
                        for part in content:
                            if part.get("type") == "text":
                                msg_lines.append(part.get("text", ""))
                            elif part.get("type") == "image_url":
                                img_counter += 1
                                msg_lines.append(f"[IMAGE {img_counter}]")
                    else:
                        msg_lines.append(content)
                messages_str = "\n".join(msg_lines)

                log_msg = "\n".join([
                    "",
                    "=" * 80,
                    "LLM SCORE (ALLOW NO ANSWER TAG)",
                    "=" * 80,
                    "Method: LLM-as-a-Judge",
                    "Answer extracted: True",
                    "EM match: False",
                    f"Golden answers: {ground_truth}",
                    f"Extracted answer: {answer}",
                    f"Solution string: {solution_str}",
                    "-" * 40,
                    "[INPUT] Messages sent to vLLM:",
                    messages_str,
                    "-" * 40,
                    "[OUTPUT] vLLM response:",
                    evaluation_text,
                    "-" * 40,
                    f"Judge decision: {judge_decision}",
                    f"Score: {llm_score}",
                    "=" * 80,
                    ""
                ])
                print(log_msg, flush=True)

    # Print summary for EM match path (LLM path already printed INPUT+OUTPUT together inside the loop)
    if do_print and scoring_method != "llm_judge":
        log_msg = "\n".join([
            "",
            "=" * 80,
            "LLM SCORE (ALLOW NO ANSWER TAG)",
            "=" * 80,
            "Method: EM",
            "Answer extracted: True",
            f"EM match: {scoring_method == 'em_match'}",
            f"Golden answers: {ground_truth}",
            f"Extracted answer: {answer}",
            f"Solution string: {solution_str}",
            f"Score: {llm_score}",
            "=" * 80,
            ""
        ])
        print(log_msg, flush=True)

    return llm_score


compute_score_fns = {
    "f1_score": compute_score_f1,
    "llm_score": compute_score_llm,
    "llm_score_allow_no_answer_tag": compute_score_llm_allow_no_answer_tag,
    "em_score": compute_score_em,
    "em_score_mcq": compute_score_em_for_mcq,
    "format_score": compute_format_score,
}
