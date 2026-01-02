# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
import copy
import json
import logging
import os
import re
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser
from verl.experimental.agent_loop.utils import add_generation_prompt_for_gpt_oss, format_gpt_oss_tool_response_manually
from verl.interactions.base import BaseInteraction
from verl.interactions.utils.interaction_registry import initialize_interactions_from_config
from verl.tools.schemas import ToolResponse
from verl.tools.utils.tool_registry import initialize_tools_from_config
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _validate_format_from_response(response_str: str) -> bool:
    """
    Validate format from decoded response string for format_score reward.
    Checks:
    1. Has <answer>...</answer> with non-empty content in final turn
    2. Tool calls (if any) are valid JSON
    """
    # Split into assistant turns (don't slice [1:] - first turn doesn't start with the token)
    assistant_turns = [t for t in response_str.split("<|im_start|>assistant") if t.strip()]

    if not assistant_turns:
        return False

    # Get last turn content (up to <|im_end|>)
    last_turn = assistant_turns[-1]
    end_pos = last_turn.find('<|im_end|>')
    if end_pos != -1:
        last_turn = last_turn[:end_pos]

    # Check for <answer> tag with non-empty content (use last match like scorer does)
    answer_matches = list(re.finditer(r'<answer>(.*?)</answer>', last_turn, re.DOTALL))
    if not answer_matches:
        return False
    answer_content = answer_matches[-1].group(1).strip()
    if not answer_content:
        return False

    # Check tool calls in non-final turns are valid JSON
    for turn in assistant_turns[:-1]:
        end_pos = turn.find('<|im_end|>')
        if end_pos != -1:
            turn = turn[:end_pos]

        tool_calls = re.findall(r'<tool_call>(.*?)</tool_call>', turn, re.DOTALL)
        for tc in tool_calls:
            try:
                json.loads(tc.strip())
            except json.JSONDecodeError:
                return False

    return True


class AgentState(Enum):
    PENDING = "pending"
    GENERATING = "generating"
    PROCESSING_TOOLS = "processing_tools"
    TERMINATED = "terminated"
    INTERACTING = "interacting"


class AgentData:
    """Encapsulates all state variables for the agent loop."""

    def __init__(
        self,
        messages: list[dict[str, Any]],
        image_data: Any,
        metrics: dict[str, Any],
        request_id: str,
        tools_kwargs: dict[str, Any],
        interaction: Optional[BaseInteraction] = None,
        interaction_kwargs: Optional[dict[str, Any]] = None,
        validate: bool = False,
        initial_prompt_ids: Optional[list[int]] = None,
    ):
        self.messages = messages
        self.image_data = image_data
        self.metrics = metrics
        self.request_id = request_id
        self.tools_kwargs = tools_kwargs
        self.interaction = interaction
        self.interaction_kwargs = interaction_kwargs or {}
        self.validate = validate  # Whether this is a validation run
        self.initial_prompt_ids = initial_prompt_ids  # Pre-computed prompt_ids from dataset

        # State variables
        self.prompt_ids: list[int] = []
        self.response_ids: list[int] = []
        self.response_mask: list[int] = []
        self.response_logprobs: list[float] = []
        self.turn_scores: list[float] = []
        self.tool_rewards: list[float] = []
        self.tool_stats: list[dict] = []  # Per-call stats for tool usage tracking
        self.user_turns = 0
        self.assistant_turns = 0

        # Temporary state for tool calls
        self.tool_calls: list[FunctionCall] = []
        self.termination_reason: str = ""


@register("tool_agent")
class ToolAgentLoop(AgentLoopBase):
    @classmethod
    def init_class(cls, config, tokenizer, processor, **kwargs):
        if cls._class_initialized:
            return
        cls._class_initialized = True
        print("Performing class-level ToolAgentLoop initialization")

        # Initialize tools from config file
        cls.tokenizer = tokenizer
        cls.processor = processor
        cls.max_user_turns = config.actor_rollout_ref.rollout.multi_turn.max_user_turns
        cls.max_assistant_turns = config.actor_rollout_ref.rollout.multi_turn.max_assistant_turns
        cls.max_parallel_calls = config.actor_rollout_ref.rollout.multi_turn.max_parallel_calls
        cls.max_tool_response_length = config.actor_rollout_ref.rollout.multi_turn.max_tool_response_length
        cls.tool_response_truncate_side = config.actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side

        # Load training tools config
        tool_config_path = config.actor_rollout_ref.rollout.multi_turn.tool_config_path
        tool_list = initialize_tools_from_config(tool_config_path) if tool_config_path else []
        cls.tools = {tool.name: tool for tool in tool_list}
        cls.tool_schemas = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list]
        print(f"Initialized training tools: {list(cls.tools.keys())}")

        # Load validation tools config (if different from training)
        val_tool_config_path = config.actor_rollout_ref.rollout.multi_turn.get("val_tool_config_path", None)
        if val_tool_config_path:
            val_tool_list = initialize_tools_from_config(val_tool_config_path)
            cls.val_tools = {tool.name: tool for tool in val_tool_list}
            cls.val_tool_schemas = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in val_tool_list]
            print(f"Initialized validation tools: {list(cls.val_tools.keys())}")
        else:
            # Use same tools for both train and val
            cls.val_tools = cls.tools
            cls.val_tool_schemas = cls.tool_schemas
            print("Using same tools for training and validation")

        cls.tool_parser = ToolParser.get_tool_parser(config.actor_rollout_ref.rollout.multi_turn.format, cls.tokenizer)
        cls.tool_parser_name = config.actor_rollout_ref.rollout.multi_turn.format

        cls.apply_chat_template_kwargs = config.data.get("apply_chat_template_kwargs", {})
        cls.prompt_length = config.actor_rollout_ref.rollout.prompt_length
        cls.response_length = config.actor_rollout_ref.rollout.response_length
        cls.system_prompt = tokenizer.apply_chat_template(
            [{}], add_generation_prompt=False, tokenize=True, **cls.apply_chat_template_kwargs
        )
        # Initialize interactions from config file
        cls.interaction_config_file = config.actor_rollout_ref.rollout.multi_turn.interaction_config_path
        if cls.interaction_config_file:
            cls.interaction_map: dict[str, BaseInteraction] = cls._initialize_interactions(cls.interaction_config_file)

    def _get_tools(self, validate: bool = False) -> dict:
        """Get the appropriate tools dict based on whether this is a validation run."""
        return self.val_tools if validate else self.tools

    def _get_tool_schemas(self, validate: bool = False) -> list:
        """Get the appropriate tool schemas based on whether this is a validation run."""
        return self.val_tool_schemas if validate else self.tool_schemas

    def _inject_image_search_into_prompt(
        self,
        messages: list,
        image_data: Optional[list],
        tools_kwargs: dict,
    ) -> tuple[list, Optional[list]]:
        """
        Inject image search results directly into the last user message.

        Instead of requiring the model to call image_search_tool(), the search results
        are pre-injected into the user prompt.

        Args:
            messages: List of message dicts
            image_data: List of images (can be None)
            tools_kwargs: Contains image_search_tool.create_kwargs with title/thumbnail lists

        Returns:
            Tuple of (modified messages, modified image_data)
        """
        from copy import deepcopy

        from PIL import Image

        from verl.tools.utils.image_utils import process_image

        # Get image search data from tools_kwargs
        image_search_kwargs = tools_kwargs.get("image_search_tool", {}).get("create_kwargs", {})
        title_list = image_search_kwargs.get("image_search_title_list", [])
        thumbnail_list = image_search_kwargs.get("image_search_thumbnail_list", [])

        if not title_list or not thumbnail_list:
            return messages, image_data

        # Deep copy messages to avoid mutating original data
        messages = deepcopy(messages)

        # Ensure image_data is a list
        if image_data is None:
            image_data = []
        else:
            image_data = list(image_data)

        # Find the last user message
        last_user_idx = -1
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user":
                last_user_idx = i
                break

        if last_user_idx < 0:
            logger.warning("No user message found for image search injection")
            return messages, image_data

        # Build image search content
        content_to_add = [
            {
                "type": "text",
                "text": "\n\nTo help you answer the question, here are reverse image search results "
                "for the given image.\n\nReverse image search results:",
            }
        ]

        for i, title in enumerate(title_list):
            # Add title text
            content_to_add.append({"type": "text", "text": f"\n\nTitle {i + 1}: {title}\nThumbnail {i + 1}: "})

            # Load and add thumbnail image
            if i < len(thumbnail_list):
                thumbnail_path = thumbnail_list[i]
                try:
                    if os.path.exists(thumbnail_path):
                        thumbnail_img = Image.open(thumbnail_path)
                        thumbnail_img = process_image(thumbnail_img)
                        # Add image placeholder to content
                        content_to_add.append({"type": "image"})
                        # Add actual image to image_data
                        image_data.append(thumbnail_img)
                    else:
                        logger.warning(f"Thumbnail not found: {thumbnail_path}")
                except Exception as e:
                    logger.warning(f"Failed to load thumbnail {thumbnail_path}: {e}")

        # Extend the last user message's content
        user_content = messages[last_user_idx].get("content", "")
        if isinstance(user_content, str):
            # Convert string content to list format
            messages[last_user_idx]["content"] = [{"type": "text", "text": user_content}] + content_to_add
        elif isinstance(user_content, list):
            # Extend existing list content
            messages[last_user_idx]["content"] = messages[last_user_idx]["content"] + content_to_add
        else:
            logger.warning(f"Unexpected content type: {type(user_content)}")

        return messages, image_data

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])

        # Load images from paths (lazy loading to reduce memory in Ray)
        multi_modal_data = kwargs.get("multi_modal_data", {})
        image_paths = multi_modal_data.get("image_paths", None)
        if image_paths:
            from verl.utils.dataset.vision_utils import process_image
            image_patch_size = self.config.data.get("image_patch_size", None)
            if image_patch_size is None:
                raise ValueError(
                    "data.image_patch_size is required. "
                    "Set to 14 for Qwen2.5-VL or 16 for Qwen3-VL."
                )
            # Convert to dict format so process_image can inject min/max pixels
            def to_image_dict(p):
                if isinstance(p, str):
                    return {"image": f"file://{p}"}
                elif isinstance(p, dict):
                    return p  # Already a dict
                else:
                    return {"image": p}  # PIL Image or other
            image_data = [process_image(to_image_dict(p), image_patch_size=image_patch_size) for p in image_paths]
        else:
            image_data = copy.deepcopy(multi_modal_data.get("image", None))

        metrics = {}
        request_id = uuid4().hex
        tools_kwargs = kwargs.get("tools_kwargs", {})

        # Inject image search results directly into user message if enabled
        directly_provide = self.config.actor_rollout_ref.rollout.multi_turn.get("directly_provide_image_search", False)
        has_image_search = tools_kwargs and "image_search_tool" in tools_kwargs
        if not hasattr(ToolAgentLoop, '_debug_printed'):
            ToolAgentLoop._debug_printed = True
            print(f"[RAG] directly_provide_image_search={directly_provide}, has_image_search_kwargs={has_image_search}", flush=True)
        if directly_provide and tools_kwargs:
            messages, image_data = self._inject_image_search_into_prompt(
                messages, image_data, tools_kwargs
            )
        validate = kwargs.get("validate", False)

        # Get pre-computed input_ids from dataset (if available)
        # Dataset's input_ids is left-padded, need to strip padding using attention_mask
        initial_prompt_ids = kwargs.get("input_ids", None)
        attention_mask = kwargs.get("attention_mask", None)
        if initial_prompt_ids is not None and attention_mask is not None:
            # Convert tensors to lists if needed
            if hasattr(initial_prompt_ids, "tolist"):
                initial_prompt_ids = initial_prompt_ids.tolist()
            if hasattr(attention_mask, "tolist"):
                attention_mask = attention_mask.tolist()
            # Strip left padding: keep only tokens where attention_mask is 1
            initial_prompt_ids = [tok for tok, mask in zip(initial_prompt_ids, attention_mask) if mask == 1]
        elif initial_prompt_ids is not None:
            # No attention_mask, can't strip padding - don't use
            initial_prompt_ids = None

        # Initialize interaction if needed
        interaction = None
        interaction_kwargs = {}
        if self.interaction_config_file:
            interaction_kwargs = kwargs["extra_info"]["interaction_kwargs"]
            if "name" not in interaction_kwargs:
                raise ValueError("'name' key is required in interaction_kwargs")
            interaction_name = interaction_kwargs["name"]
            if interaction_name not in self.interaction_map:
                raise ValueError(
                    f"Interaction '{interaction_name}' not found in interaction_map. Available interactions: "
                    f"{list(self.interaction_map.keys())}"
                )
            interaction = self.interaction_map[interaction_name]
            await interaction.start_interaction(request_id, **interaction_kwargs)
        # Create AgentData instance to encapsulate all state
        agent_data = AgentData(
            messages=messages,
            image_data=image_data,
            metrics=metrics,
            request_id=request_id,
            tools_kwargs=tools_kwargs,
            interaction=interaction,
            interaction_kwargs=interaction_kwargs,
            validate=validate,
            initial_prompt_ids=initial_prompt_ids,
        )

        # State machine loop
        state = AgentState.PENDING
        while state != AgentState.TERMINATED:
            if state == AgentState.PENDING:
                state = await self._handle_pending_state(agent_data, sampling_params)
            elif state == AgentState.GENERATING:
                state = await self._handle_generating_state(agent_data, sampling_params)
            elif state == AgentState.PROCESSING_TOOLS:
                state = await self._handle_processing_tools_state(agent_data)
            elif state == AgentState.INTERACTING:
                state = await self._handle_interacting_state(agent_data)
            else:
                logger.error(f"Invalid state: {state}")
                state = AgentState.TERMINATED

        # Finalize output
        # Handle edge case where response_mask is empty (e.g., prompt_too_long on first turn)
        if len(agent_data.response_mask) == 0:
            response_ids = []
            prompt_ids = list(agent_data.prompt_ids)
        else:
            response_ids = agent_data.prompt_ids[-len(agent_data.response_mask) :]
            prompt_ids = agent_data.prompt_ids[: len(agent_data.prompt_ids) - len(agent_data.response_mask)]
        multi_modal_data = {"image": agent_data.image_data} if agent_data.image_data is not None else {}

        # Validate format for format_score reward (use truncated response_ids to match what's returned)
        truncated_response_ids = response_ids[: self.response_length]
        response_str = self.tokenizer.decode(truncated_response_ids)
        is_valid_format = _validate_format_from_response(response_str)

        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=agent_data.response_mask[: self.response_length],
            multi_modal_data=multi_modal_data,
            response_logprobs=agent_data.response_logprobs[: self.response_length]
            if agent_data.response_logprobs
            else None,
            num_turns=agent_data.user_turns + agent_data.assistant_turns + 1,
            metrics=agent_data.metrics,
            extra_fields={},
        )
        output.extra_fields.update({
            "turn_scores": agent_data.turn_scores,
            "tool_rewards": agent_data.tool_rewards,
            "tool_stats": agent_data.tool_stats,
            "is_valid_format": is_valid_format,
        })

        return output

    async def _handle_pending_state(self, agent_data: AgentData, sampling_params: dict[str, Any]) -> AgentState:
        """Handle the pending state: prepare the prompt and start generation."""
        # Reuse pre-computed input_ids from dataset if available (for training, not validation)
        if agent_data.initial_prompt_ids is not None and not agent_data.validate:
            agent_data.prompt_ids = list(agent_data.initial_prompt_ids)  # Copy to avoid mutating original
            return AgentState.GENERATING

        # Otherwise compute prompt_ids (for validation or when dataset didn't provide input_ids)
        tool_schemas = self._get_tool_schemas(agent_data.validate)
        if self.processor is not None:
            raw_prompt = await self.loop.run_in_executor(
                None,
                lambda: self.processor.apply_chat_template(
                    agent_data.messages,
                    tools=tool_schemas,
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                ),
            )
            model_inputs = self.processor(text=[raw_prompt], images=agent_data.image_data, return_tensors="pt")
            agent_data.prompt_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
        else:
            agent_data.prompt_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    agent_data.messages,
                    tools=tool_schemas,
                    add_generation_prompt=True,
                    tokenize=True,
                    **self.apply_chat_template_kwargs,
                ),
            )
        return AgentState.GENERATING

    async def _handle_generating_state(
        self, agent_data: AgentData, sampling_params: dict[str, Any], ignore_termination: bool = False
    ) -> AgentState:
        """Handle the generating state: generate model response and check for tool calls."""
        add_messages: list[dict[str, Any]] = []

        try:
            with simple_timer("generate_sequences", agent_data.metrics):
                output = await self.server_manager.generate(
                    request_id=agent_data.request_id,
                    prompt_ids=agent_data.prompt_ids,
                    sampling_params=sampling_params,
                    image_data=agent_data.image_data,
                )
        except Exception as e:
            error_msg = str(e)
            if "Multimodal prompt is too long" in error_msg:
                logger.warning(f"Prompt too long for request {agent_data.request_id}, terminating gracefully: {error_msg}")
                agent_data.termination_reason = "prompt_too_long"
                agent_data.response_ids = []
                return AgentState.TERMINATED
            else:
                raise  # Re-raise other errors

        agent_data.assistant_turns += 1
        agent_data.response_ids = output.token_ids
        agent_data.prompt_ids += agent_data.response_ids
        agent_data.response_mask += [1] * len(agent_data.response_ids)
        if output.log_probs:
            agent_data.response_logprobs += output.log_probs

        # Check termination conditions
        termination_reason = None
        if not ignore_termination and len(agent_data.response_mask) >= self.response_length:
            termination_reason = "max_response_length"
        elif self.max_assistant_turns and agent_data.assistant_turns >= self.max_assistant_turns:
            termination_reason = "max_assistant_turns"
        elif self.max_user_turns and agent_data.user_turns >= self.max_user_turns:
            termination_reason = "max_user_turns"

        # Extract tool calls
        _, agent_data.tool_calls = await self.tool_parser.extract_tool_calls(agent_data.response_ids)

        if termination_reason:
            agent_data.termination_reason = termination_reason
            return AgentState.TERMINATED

        # Handle interaction if needed
        if self.interaction_config_file:
            assistant_message = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=True)
            )
            add_messages.append({"role": "assistant", "content": assistant_message})
            agent_data.messages.extend(add_messages)

        # Determine next state
        if agent_data.tool_calls:
            return AgentState.PROCESSING_TOOLS
        elif self.interaction_config_file:
            return AgentState.INTERACTING
        else:
            agent_data.termination_reason = "no_tool_calls"
            return AgentState.TERMINATED

    async def _handle_processing_tools_state(self, agent_data: AgentData) -> AgentState:
        """Handle the processing tools state: execute tool calls and prepare tool responses."""
        add_messages: list[dict[str, Any]] = []
        new_images_this_turn: list[Any] = []  # Local variable instead of agent_data attribute

        tasks = []
        tool_call_names = []
        for tool_call in agent_data.tool_calls[: self.max_parallel_calls]:
            tasks.append(self._call_tool(tool_call, agent_data.tools_kwargs, agent_data.image_data, agent_data.validate))
            tool_call_names.append(tool_call.name)

        with simple_timer("tool_calls", agent_data.metrics):
            responses = await asyncio.gather(*tasks)

        # Process tool responses and update multi_modal_data
        # Removed: agent_data.new_images_this_turn = []
        for tool_response, tool_reward, tool_stat in responses:
            # Collect tool stats for metrics tracking
            agent_data.tool_stats.append(tool_stat)
            # Create message from tool response
            if tool_response.has_interleaved_content():
                # Handle interleaved text/image content (e.g., image_search results)
                # This produces: text -> image -> text -> image -> ... -> text
                if not getattr(self.processor, "image_processor", None):
                    raise ValueError(
                        "Multimedia data can only be processed by `processor`, but the processor is None. "
                        "This error is often caused if you are using a LLM model but your tool returns multimodal "
                        "data. Plase use a vlm as the base model."
                    )
                content = []
                for item in tool_response.interleaved_content:
                    if item.get("type") == "text":
                        content.append({"type": "text", "text": item.get("text", "")})
                    elif item.get("type") == "image":
                        content.append({"type": "image"})
                        # Collect image for processing
                        img = item.get("image")
                        if img is not None:
                            new_images_this_turn.append(img)
                    elif item.get("type") == "video":
                        content.append({"type": "video"})
                        logger.warning("Multimedia type 'video' is not currently supported.")
                        raise NotImplementedError("Multimedia type 'video' is not currently supported.")
                message = {"role": "tool", "content": content}
            elif tool_response.image or tool_response.video:
                # Multi-modal content with structured format
                # Order: text (before) -> image/video -> text_after_image
                if not getattr(self.processor, "image_processor", None):
                    raise ValueError(
                        "Multimedia data can only be processed by `processor`, but the processor is None. "
                        "This error is often caused if you are using a LLM model but your tool returns multimodal "
                        "data. Plase use a vlm as the base model."
                    )
                content = []
                # Add text before image (if present)
                if tool_response.text:
                    content.append({"type": "text", "text": tool_response.text})
                # Add image/video - add one marker per image!
                if tool_response.image:
                    for _ in tool_response.image:
                        content.append({"type": "image"})
                if tool_response.video:
                    for _ in tool_response.video:
                        content.append({"type": "video"})
                # Add text after image (if present)
                if tool_response.text_after_image:
                    content.append({"type": "text", "text": tool_response.text_after_image})
                message = {"role": "tool", "content": content}

                # Collect images for processing
                if tool_response.image:
                    for img in tool_response.image:
                        if img is not None:
                            new_images_this_turn.append(img)

                # Handle video data
                if tool_response.video:
                    logger.warning("Multimedia type 'video' is not currently supported. Only 'image' is supported.")
                    raise NotImplementedError(
                        "Multimedia type 'video' is not currently supported. Only 'image' is supported."
                    )
            else:
                # Text-only content
                message = {"role": "tool", "content": tool_response.text or ""}

            add_messages.append(message)

            if tool_reward is not None:
                agent_data.tool_rewards.append(tool_reward)

        agent_data.messages.extend(add_messages)

        # Update prompt with tool responses
        if self.processor is not None:
            raw_tool_response = await self.loop.run_in_executor(
                None,
                lambda: self.processor.apply_chat_template(
                    add_messages,
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                ),
            )
            # Use only the new images from this turn for processing tool responses
            current_images = new_images_this_turn if new_images_this_turn else None  # Using local variable
            model_inputs = self.processor(text=[raw_tool_response], images=current_images, return_tensors="pt")
            response_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
        else:
            if self.tool_parser_name == "gpt-oss":
                logger.info("manually format tool responses for gpt-oss")
                # Format tool responses manually
                tool_response_texts = []
                for i, tool_msg in enumerate(add_messages):
                    actual_tool_name = tool_call_names[i]
                    formatted = format_gpt_oss_tool_response_manually(tool_msg["content"], actual_tool_name)
                    tool_response_texts.append(formatted)

                tool_response_text = add_generation_prompt_for_gpt_oss("".join(tool_response_texts))
                response_ids = await self.loop.run_in_executor(
                    None, lambda: self.tokenizer.encode(tool_response_text, add_special_tokens=False)
                )
            else:
                response_ids = await self.loop.run_in_executor(
                    None,
                    lambda: self.tokenizer.apply_chat_template(add_messages, add_generation_prompt=True, tokenize=True),
                )
                response_ids = response_ids[len(self.system_prompt) :]
        if len(agent_data.response_mask) + len(response_ids) >= self.response_length:
            return AgentState.TERMINATED
        # Update prompt_ids and response_mask

        if new_images_this_turn:
            if agent_data.image_data is None:
                agent_data.image_data = []
            elif not isinstance(agent_data.image_data, list):
                agent_data.image_data = [agent_data.image_data]
            for img in new_images_this_turn:
                agent_data.image_data.append(img)

        agent_data.prompt_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)
        agent_data.user_turns += 1
        return AgentState.GENERATING

    async def _handle_interacting_state(self, agent_data: AgentData) -> AgentState:
        """Handle the interacting state: get user input from interaction."""
        (
            should_terminate_sequence,
            interaction_responses,
            reward,
            metrics,
        ) = await agent_data.interaction.generate_response(
            agent_data.request_id, agent_data.messages, **agent_data.interaction_kwargs
        )
        agent_data.user_turns += 1

        add_messages: list[dict[str, Any]] = [{"role": "user", "content": interaction_responses}]
        agent_data.messages.extend(add_messages)

        if reward is not None:
            agent_data.turn_scores.append(reward)

        # Update prompt with user responses (similar to _handle_processing_tools_state)
        if self.processor is not None:
            raw_user_response = await self.loop.run_in_executor(
                None,
                lambda: self.processor.apply_chat_template(
                    add_messages,
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                ),
            )
            model_inputs = self.processor(text=[raw_user_response], images=None, return_tensors="pt")
            response_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
        else:
            response_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(add_messages, add_generation_prompt=True, tokenize=True),
            )
        response_ids = response_ids[len(self.system_prompt) :]

        # Update prompt_ids and response_mask
        agent_data.prompt_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)

        # double check prompt
        # Check termination condition
        if should_terminate_sequence:
            return AgentState.TERMINATED
        else:
            return AgentState.GENERATING

    async def _call_tool(
        self, tool_call: FunctionCall, tools_kwargs: dict[str, Any], images: Optional[list[Any]] = None,
        validate: bool = False
    ) -> tuple[ToolResponse, float, dict]:
        """Call tool and return tool response.

        Args:
            tool_call: The function call to execute
            tools_kwargs: Tool-specific kwargs from dataset
            images: List of images from conversation history (for tools like image_crop_tool)
            validate: Whether this is a validation run (uses val_tools if True)

        Returns:
            tuple: (ToolResponse, tool_reward, stats_dict) where stats_dict contains:
                - tool_name: Name of the tool called
                - success: Whether the call succeeded
                - error_type: One of None, 'invalid_json', 'unknown_tool', 'execution_error'
        """
        # Initialize stats for this call
        stats = {
            "tool_name": tool_call.name,
            "success": False,
            "error_type": None,
        }

        tool, instance_id = None, None
        # Use appropriate tools dict based on validate flag
        tools = self._get_tools(validate)

        # Parse arguments
        try:
            tool_args = json.loads(tool_call.arguments)
        except json.JSONDecodeError as e:
            stats["error_type"] = "invalid_json"
            logger.warning(f"Invalid JSON in tool arguments: {e}")
            return (
                ToolResponse(text=f"Error: Invalid JSON in tool arguments: {e}"),
                0.0,
                stats,
            )

        # Check tool exists
        if tool_call.name not in tools:
            stats["error_type"] = "unknown_tool"
            logger.warning(f"Unknown tool: {tool_call.name}")
            return (
                ToolResponse(text=f"Error: Unknown tool '{tool_call.name}'"),
                0.0,
                stats,
            )

        # Execute tool
        try:
            tool = tools[tool_call.name]
            kwargs = tools_kwargs.get(tool_call.name, {})
            # Pass images to create() for tools that need them (e.g., image_crop_tool)
            create_kwargs = kwargs.get("create_kwargs", {})
            if images is not None:
                create_kwargs["images"] = images
            instance_id, _ = await tool.create(create_kwargs=create_kwargs)
            # Pass images to execute() as well for tools that might need dynamic access
            tool_execution_response, tool_reward, res = await tool.execute(instance_id, tool_args, images=images)
            stats["success"] = True
        except RuntimeError:
            # Re-raise RuntimeError to crash training (e.g., text_search server down)
            raise
        except Exception as e:
            stats["error_type"] = "execution_error"
            logger.warning(f"Error when executing tool: {e}")
            return (
                ToolResponse(text=f"Error when executing tool: {e}"),
                0.0,
                stats,
            )
        finally:
            if tool and instance_id:
                await tool.release(instance_id)

        tool_response_text = tool_execution_response.text
        if tool_response_text and len(tool_response_text) > self.max_tool_response_length:
            if self.tool_response_truncate_side == "left":
                tool_response_text = tool_response_text[: self.max_tool_response_length] + "...(truncated)"
            elif self.tool_response_truncate_side == "right":
                tool_response_text = "(truncated)..." + tool_response_text[-self.max_tool_response_length :]
            else:
                length = self.max_tool_response_length // 2
                tool_response_text = tool_response_text[:length] + "...(truncated)..." + tool_response_text[-length:]

        # Create ToolResponse from tool execution result
        tool_response_kwargs = {"text": tool_response_text}

        # Add multimedia data if present
        for attr_name in ["image", "video", "text_after_image", "interleaved_content"]:
            if hasattr(tool_execution_response, attr_name):
                attr_value = getattr(tool_execution_response, attr_name)
                if attr_value is not None:
                    tool_response_kwargs[attr_name] = attr_value

        return ToolResponse(**tool_response_kwargs), tool_reward, stats

    @classmethod
    def _initialize_interactions(cls, interaction_config_file):
        """Initialize interactions from configuration.
        Returns:
            dict[str, BaseInteraction]: A dictionary mapping interaction names to interaction instances.
        """
        if interaction_config_file is None:
            return {}

        interaction_map = initialize_interactions_from_config(interaction_config_file)
        logger.info(f"Initialize interactions from configuration: interaction_map: {list(interaction_map.keys())}")
        return interaction_map
