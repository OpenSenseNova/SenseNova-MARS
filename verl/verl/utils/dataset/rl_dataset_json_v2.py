# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
RLHF Dataset for JSONL files with tool training support.

Based on rl_dataset.py with additional features for JSONL format and tool training.
Supports JSONL format with meta.json configuration.
"""

import copy
import json
import logging
import os
import random
import re
import traceback
from collections import defaultdict
from typing import Optional

import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask

logger = logging.getLogger(__name__)

# Global variable for multiprocessing worker initialization
_filter_worker_state = {}


def _init_filter_worker(processor_class, processor_config, tokenizer_class, tokenizer_config,
                        prompt_key, image_key, video_key, image_patch_size, max_prompt_length,
                        apply_chat_template_kwargs, tool_schemas):
    """Initialize worker process with processor and tokenizer."""
    global _filter_worker_state
    # Reconstruct processor and tokenizer in worker process
    if processor_class is not None:
        _filter_worker_state['processor'] = processor_class.from_pretrained(**processor_config)
    else:
        _filter_worker_state['processor'] = None
    _filter_worker_state['tokenizer'] = tokenizer_class.from_pretrained(**tokenizer_config)
    _filter_worker_state['prompt_key'] = prompt_key
    _filter_worker_state['image_key'] = image_key
    _filter_worker_state['video_key'] = video_key
    _filter_worker_state['image_patch_size'] = image_patch_size
    _filter_worker_state['max_prompt_length'] = max_prompt_length
    _filter_worker_state['apply_chat_template_kwargs'] = apply_chat_template_kwargs
    _filter_worker_state['tool_schemas'] = tool_schemas


def _filter_check_doc(args):
    """Check if document length is within limit. Runs in worker process."""
    idx, doc = args
    global _filter_worker_state

    processor = _filter_worker_state['processor']
    tokenizer = _filter_worker_state['tokenizer']
    prompt_key = _filter_worker_state['prompt_key']
    image_key = _filter_worker_state['image_key']
    video_key = _filter_worker_state['video_key']
    image_patch_size = _filter_worker_state['image_patch_size']
    max_prompt_length = _filter_worker_state['max_prompt_length']
    apply_chat_template_kwargs = _filter_worker_state['apply_chat_template_kwargs']
    tool_schemas = _filter_worker_state['tool_schemas']

    try:
        doc_copy = copy.deepcopy(doc)
        messages = doc_copy.pop(prompt_key)

        # Get input_template arguments and inject system_prompt/format_instruction
        input_template = doc_copy.get('input_template', {})
        kwargs = input_template.get('arguments', {})

        if kwargs.get('system_prompt'):
            for m in messages:
                if m['role'] == 'system':
                    raise ValueError("system prompt already exists in messages")
            messages.insert(0, {'role': 'system', 'content': kwargs['system_prompt']})

        if kwargs.get('format_instruction'):
            for m in messages:
                if m['role'] == 'user':
                    m['content'] = m['content'].rstrip() + '\n' + kwargs['format_instruction']
                    break

        # Convert <image>/<video> to HF format
        if image_key in doc or video_key in doc:
            for message in messages:
                content = message["content"]
                if isinstance(content, str):
                    content_list = []
                    segments = re.split("(<image>|<video>)", content)
                    segments = [item for item in segments if item != ""]
                    for segment in segments:
                        if segment == "<image>":
                            content_list.append({"type": "image"})
                        elif segment == "<video>":
                            content_list.append({"type": "video"})
                        else:
                            content_list.append({"type": "text", "text": segment})
                    message["content"] = content_list

        apply_kwargs = dict(**apply_chat_template_kwargs)
        if tool_schemas is not None:
            apply_kwargs["tools"] = tool_schemas

        if processor is not None:
            from verl.utils.dataset.vision_utils import process_image, process_video

            raw_prompt = processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False, **apply_kwargs
            )

            if image_key in doc and doc[image_key]:
                images = [process_image(image, image_patch_size=image_patch_size) for image in doc[image_key]]
            else:
                images = None

            if video_key in doc and doc[video_key]:
                videos, video_metadata = zip(
                    *[process_video(video, image_patch_size=image_patch_size, return_video_metadata=True)
                      for video in doc[video_key]],
                    strict=True,
                )
                videos = list(videos)
                videos_kwargs = {"video_metadata": list(video_metadata), "do_sample_frames": False}
            else:
                videos = None
                videos_kwargs = {}

            length = len(processor(text=[raw_prompt], images=images, videos=videos, videos_kwargs=videos_kwargs)["input_ids"][0])
        else:
            length = len(tokenizer.apply_chat_template(messages, add_generation_prompt=True, **apply_kwargs))

        return idx, doc, length <= max_prompt_length
    except Exception as e:
        print(f"Error checking doc {idx}: {e}")
        traceback.print_exc()
        return idx, doc, False


def load_local_jsonl(file_path: str) -> list:
    """Load a JSONL file and return list of dicts."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def collate_fn(data_list: list[dict]) -> dict:
    """
    Collate a batch of sample dicts into batched tensors and arrays.

    Args:
        data_list: List of dicts mapping feature names to torch.Tensor or other values.

    Returns:
        Dict where tensor entries are stacked into a torch.Tensor of shape
        (batch_size, \\*dims) and non-tensor entries are converted to
        np.ndarray of dtype object with shape (batch_size,).
    """
    # Collect all keys across all samples to handle optional fields
    # (e.g., mean8_llm_score exists in some samples but not others)
    all_keys = set()
    for data in data_list:
        all_keys.update(data.keys())

    tensors = defaultdict(list)
    non_tensors = defaultdict(list)

    for data in data_list:
        for key in all_keys:
            val = data.get(key, None)  # Use None for missing optional fields
            if isinstance(val, torch.Tensor):
                tensors[key].append(val)
            else:
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.fromiter(val, dtype=object, count=len(val))

    return {**tensors, **non_tensors}


class RLHFJSONDatasetV2(Dataset):
    """
    RLHF Dataset for JSONL files with tool training support.

    Based on RLHFDataset from rl_dataset.py with additional features:
    - JSONL format loading via meta.json configuration
    - Tool training specific config (image_search metadata)
    - input_template handling (system_prompt, format_instruction injection)
    - reward_model configuration from meta.json
    - repeat_time for data augmentation
    - agent_name for tool training

    Output format for AgentLoop:
        - raw_prompt: HF format messages [{"role": "user", "content": [{"type": "image"}, ...]}]
        - multi_modal_data["image_paths"]: Image path dicts [{"image": path}, ...] (lazy loaded in agent loop)

    Args:
        data_files (str): Path to meta.json file.
        tokenizer (PreTrainedTokenizer): For the tokenization of text to token IDs.
        config (DictConfig): Options like prompt_key, max_prompt_length, truncation, etc.
        processor (ProcessorMixin, optional): Multimodal preprocessor for images/videos.
        max_samples (int): Maximum number of samples to use (-1 for all).
    """

    def __init__(
        self,
        data_files: str,
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
        max_samples: int = -1,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_samples = max_samples
        self.config = config

        # Validate meta file exists
        assert os.path.exists(data_files), f'meta_file {data_files} not found'
        self.meta_file = data_files
        self.original_meta_file = copy.deepcopy(data_files)  # for resume
        with open(data_files, 'r') as f:
            self.meta_info = json.load(f)

        # Config options (aligned with rl_dataset.py)
        self.prompt_key = config.get("prompt_key", "prompt")
        self.image_key = config.get("image_key", "images")
        self.video_key = config.get("video_key", "videos")
        if "image_patch_size" not in config:
            raise ValueError(
                "data.image_patch_size is required. "
                "Set to 14 for Qwen2.5-VL or 16 for Qwen3-VL."
            )
        self.image_patch_size = config.image_patch_size
        self.max_prompt_length = config.get("max_prompt_length", 1024)
        self.return_raw_chat = config.get("return_raw_chat", False)
        self.return_full_prompt = config.get("return_full_prompt", False)
        self.truncation = config.get("truncation", "error")
        self.filter_overlong_prompts = config.get("filter_overlong_prompts", False)
        self.apply_chat_template_kwargs = config.get("apply_chat_template_kwargs", {})

        # Tool config from rl_dataset.py
        # Support separate tool configs for train and val datasets
        # This is determined AFTER checking is_train below, so we store both paths first
        self._tool_config_path = config.get("tool_config_path", None)
        self._val_tool_config_path = config.get("val_tool_config_path", None)
        self.tool_config_path = None  # Will be set after is_train is determined
        self.tool_schemas = None

        self.num_workers = config.get("filter_overlong_prompts_workers", max(1, os.cpu_count() // 4))
        self.num_workers = min(self.num_workers, os.cpu_count()) if self.num_workers is not None else None
        self.chat_template_func = config.get("chat_template_func", None)
        self.need_tools_kwargs = config.get("need_tools_kwargs", False)
        self.filter_prompts = config.get("filter_prompts", True)
        self.serialize_dataset = False
        self.return_multi_modal_inputs = config.get("return_multi_modal_inputs", True)
        self.seed = config.get("seed")

        # Tool training specific config
        self.image_search_title_list_key = config.get("image_search_title_list_key", "image_search_title_list")
        self.image_search_thumbnail_list_key = config.get("image_search_thumbnail_list_key", "image_search_thumbnail_list")
        self.image_search_summary_key = config.get("image_search_summary_key", "image_search_summary")
        self.image_search_max_results = config.get("image_search_max_results", 5)

        # Determine train/test from filename
        filename = os.path.basename(self.meta_file)
        if "train" in filename:
            self.is_train = True
        elif "test" in filename:
            self.is_train = False
        else:
            raise ValueError(f"filename '{filename}' must contain 'train' or 'test'")

        # Now that is_train is determined, select the appropriate tool config
        # For validation datasets, use val_tool_config_path (no fallback to avoid silent bugs)
        if self.is_train:
            self.tool_config_path = self._tool_config_path
        else:
            self.tool_config_path = self._val_tool_config_path

        # Initialize tool schemas from the selected config
        if self.tool_config_path:
            try:
                from verl.tools.utils.tool_registry import initialize_tools_from_config

                tool_list = initialize_tools_from_config(self.tool_config_path)
                self.tool_schemas = [
                    tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list
                ]
                logger.info(f"Initialized tool schemas from {self.tool_config_path} for {'train' if self.is_train else 'val'} dataset")
            except Exception as e:
                logger.warning("Failed to initialize tools from %s: %s", self.tool_config_path, e)
                self.tool_schemas = None

        self.shuffle = config.get("shuffle", self.is_train)

        self._read_files_and_tokenize()

    def _read_files_and_tokenize(self, in_worker=False):
        """Load JSONL data with metadata from meta.json.

        Args:
            in_worker: If True, this is called from a DataLoader worker process.
                      Filtering will be done sequentially to avoid nested multiprocessing.
        """
        # Seed random for reproducibility (important for DataLoader workers)
        if self.seed is not None:
            random.seed(self.seed)

        self.dataframe = []

        for dataset_name, dataset_info in self.meta_info.items():
            # Load JSONL data
            data_list = load_local_jsonl(dataset_info['annotation'])
            expected_len = dataset_info['length']
            assert len(data_list) == expected_len, \
                f"dataset {dataset_name}: got {len(data_list)} samples, expected {expected_len}"

            # Handle repeat_time (data augmentation)
            repeat_time = dataset_info.get('repeat_time', 1.0)
            full_repeats = data_list * int(repeat_time)
            partial = repeat_time - int(repeat_time)
            if partial > 0:
                extra_count = int(len(data_list) * partial)
                extra_samples = random.sample(data_list, extra_count)
                data_list = full_repeats + extra_samples
            else:
                data_list = full_repeats

            # Process each sample
            root = dataset_info.get('root')
            for i, d in enumerate(data_list):
                d = d.copy()

                # Add dataset-level info
                d['data_source'] = dataset_name
                d['id'] = f"{dataset_name}-{d.get('index', i)}"

                # Resolve image paths and convert to dict format for vision_utils.process_image
                if self.image_key in d:
                    images = d[self.image_key]
                    if not isinstance(images, list):
                        images = [images]
                    if root:
                        d[self.image_key] = [{"image": os.path.join(root, img)} for img in images]
                    else:
                        d[self.image_key] = [{"image": img} for img in images]

                # Resolve video paths and convert to dict format for vision_utils.process_video
                if self.video_key in d:
                    videos = d[self.video_key]
                    if not isinstance(videos, list):
                        videos = [videos]
                    if root:
                        d[self.video_key] = [{"video": os.path.join(root, vid)} for vid in videos]
                    else:
                        d[self.video_key] = [{"video": vid} for vid in videos]

                # Resolve thumbnail paths
                if root and self.image_search_thumbnail_list_key in d:
                    thumbnails = d[self.image_search_thumbnail_list_key]
                    if isinstance(thumbnails, list):
                        d[self.image_search_thumbnail_list_key] = [
                            os.path.join(root, t) for t in thumbnails
                        ]

                # Add dataset-level config
                assert 'input_template' in dataset_info, \
                    f"dataset {dataset_name}: 'input_template' required in meta.json"
                d['input_template'] = dataset_info['input_template']

                # Add reward config
                assert 'reward_fn' in dataset_info, \
                    f"dataset {dataset_name}: 'reward_fn' required in meta.json"
                assert 'unused_reward_fn' in dataset_info, \
                    f"dataset {dataset_name}: 'unused_reward_fn' required in meta.json"

                if 'reward_model' not in d:
                    d['reward_model'] = {}
                d['reward_model']['reward_fn'] = dataset_info['reward_fn']
                d['reward_model']['unused_reward_fn'] = dataset_info['unused_reward_fn']

                # Validate training data has reward functions
                if self.is_train and not dataset_info['reward_fn']:
                    raise ValueError(
                        f"Training dataset '{dataset_name}' has empty reward_fn. "
                        f"Please add at least one reward function."
                    )

                self.dataframe.append(d)

        total = len(self.dataframe)
        print(f"dataset len: {len(self.dataframe)}")

        # Apply max_samples limit
        if self.max_samples > 0 and self.max_samples < total:
            if self.shuffle:
                rngs_args = (self.seed,) if self.seed is not None else ()
                rng = np.random.default_rng(*rngs_args)
                indices = rng.choice(total, size=self.max_samples, replace=False)
                self.dataframe = [self.dataframe[i] for i in indices]
            else:
                self.dataframe = self.dataframe[:self.max_samples]
            print(f"selected {self.max_samples} samples out of {total}")

        # Filter long prompts (workers use sequential processing to avoid nested multiprocessing)
        self.dataframe = self.maybe_filter_out_long_prompts(
            self.dataframe, use_multiprocessing=not in_worker
        )

    def maybe_filter_out_long_prompts(self, dataframe: list, use_multiprocessing: bool = True):
        """Filter out prompts that exceed max_prompt_length.

        Args:
            dataframe: List of documents to filter.
            use_multiprocessing: If True, use multiprocessing Pool for parallel filtering.
                                If False, process sequentially (for DataLoader workers to avoid nested multiprocessing).
        """
        if not self.filter_overlong_prompts:
            return dataframe

        original_len = len(dataframe)

        # Prepare initializer arguments
        processor_class = type(self.processor) if self.processor is not None else None
        if self.processor is not None:
            processor_name = getattr(self.processor, 'name_or_path', None) or getattr(self.processor, 'model_name_or_path', None)
            processor_config = {"pretrained_model_name_or_path": processor_name}
        else:
            processor_config = None
        tokenizer_class = type(self.tokenizer)
        tokenizer_name = getattr(self.tokenizer, 'name_or_path', None) or getattr(self.tokenizer, 'model_name_or_path', None)
        tokenizer_config = {"pretrained_model_name_or_path": tokenizer_name}

        init_args = (
            processor_class, processor_config,
            tokenizer_class, tokenizer_config,
            self.prompt_key, self.image_key, self.video_key,
            self.image_patch_size, self.max_prompt_length,
            dict(self.apply_chat_template_kwargs), self.tool_schemas
        )

        if use_multiprocessing:
            from multiprocessing import Pool

            print(f"Filtering {original_len} prompts with {self.num_workers} workers...")

            # Use multiprocessing Pool with initializer
            with Pool(processes=self.num_workers, initializer=_init_filter_worker, initargs=init_args) as pool:
                # Process in chunks for progress reporting
                chunk_size = max(1, original_len // 100)
                results = []
                for i, result in enumerate(pool.imap(_filter_check_doc, enumerate(dataframe), chunksize=chunk_size)):
                    results.append(result)
                    if (i + 1) % 1000 == 0:
                        print(f"Filtering progress: {i + 1}/{original_len}")
        else:
            # Sequential processing (for DataLoader workers to avoid nested multiprocessing)
            print(f"Filtering {original_len} prompts sequentially...")

            # Initialize worker state directly
            _init_filter_worker(*init_args)

            results = []
            for i, doc in enumerate(dataframe):
                result = _filter_check_doc((i, doc))
                results.append(result)
                if (i + 1) % 1000 == 0:
                    print(f"Filtering progress: {i + 1}/{original_len}")

        # Sort by index to maintain order and extract filtered docs
        results.sort(key=lambda x: x[0])
        filtered = [doc for idx, doc, keep in results if keep]

        print(f"filter dataset len: {len(filtered)} (filtered {original_len - len(filtered)} samples)")
        return filtered

    def resume_dataset_state(self):
        """Resume dataset state from checkpoint."""
        self.serialize_dataset = not hasattr(self, "original_meta_file")
        if not self.serialize_dataset:
            self._read_files_and_tokenize()
        else:
            print(r"old dataloader ckpt file is used, please train from scratch for better ckpt performance")

    def __len__(self):
        return len(self.dataframe)

    def _build_messages(self, row_dict: dict) -> list:
        """
        Build HF format messages from raw chat.

        Applies:
        1. system_prompt injection (if provided in input_template)
        2. format_instruction injection (if provided in input_template)
        3. <image>/<video> to {"type": "image"}/{"type": "video"} conversion

        Returns HF format messages compatible with processor.apply_chat_template().
        """
        messages: list = row_dict.pop(self.prompt_key)

        # Get input_template arguments
        input_template = row_dict.get('input_template', {})
        kwargs = input_template.get('arguments', {})

        # Inject system_prompt
        if kwargs.get('system_prompt'):
            # Ensure no existing system message
            for m in messages:
                if m['role'] == 'system':
                    raise ValueError("system prompt already exists in messages")
            messages.insert(0, {'role': 'system', 'content': kwargs['system_prompt']})

        # Inject format_instruction to first user message
        if kwargs.get('format_instruction'):
            for m in messages:
                if m['role'] == 'user':
                    m['content'] = m['content'].rstrip() + '\n' + kwargs['format_instruction']
                    break

        # Convert <image>/<video> to HF format (aligned with rl_dataset.py)
        if self.image_key in row_dict or self.video_key in row_dict:
            for message in messages:
                content = message["content"]
                if isinstance(content, str):
                    content_list = []
                    segments = re.split("(<image>|<video>)", content)
                    segments = [item for item in segments if item != ""]
                    for segment in segments:
                        if segment == "<image>":
                            content_list.append({"type": "image"})
                        elif segment == "<video>":
                            content_list.append({"type": "video"})
                        else:
                            content_list.append({"type": "text", "text": segment})
                    message["content"] = content_list

        return messages

    def __getitem__(self, item):
        """Returns preprocessed data for the given index."""
        row_dict: dict = copy.deepcopy(self.dataframe[item])
        messages = self._build_messages(row_dict)
        model_inputs = {}

        if self.processor is not None:
            from verl.utils.dataset.vision_utils import process_image, process_video

            apply_kwargs = dict(self.apply_chat_template_kwargs)
            if self.tool_schemas is not None:
                apply_kwargs["tools"] = self.tool_schemas
            raw_prompt = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False, **apply_kwargs
            )
            multi_modal_data = {}

            images = None
            row_dict_images = row_dict.pop(self.image_key, None)
            if row_dict_images:
                # Load images temporarily for processor() to compute correct input_ids
                images = [process_image(image, image_patch_size=self.image_patch_size) for image in row_dict_images]
                # Store paths instead of PIL images to reduce memory (agent loop will reload)
                multi_modal_data["image_paths"] = row_dict_images  # [{"image": path}, ...]

            videos = None
            videos_kwargs = {}
            row_dict_videos = row_dict.pop(self.video_key, None)
            if row_dict_videos:
                videos, video_metadata = zip(
                    *[
                        process_video(video, image_patch_size=self.image_patch_size, return_video_metadata=True)
                        for video in row_dict_videos
                    ],
                    strict=True,
                )
                videos = list(videos)
                video_metadata = list(video_metadata)
                videos_kwargs = {"video_metadata": video_metadata, "do_sample_frames": False}
                # Store paths instead of video data to reduce memory
                multi_modal_data["video_paths"] = row_dict_videos  # [{"video": path}, ...]

            model_inputs = self.processor(
                text=[raw_prompt], images=images, videos=videos, videos_kwargs=videos_kwargs, return_tensors="pt"
            )

            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

            if "second_per_grid_ts" in model_inputs:
                model_inputs.pop("second_per_grid_ts")

            row_dict["multi_modal_data"] = multi_modal_data
            # Note: images/videos are discarded here (not stored), only paths remain

            if self.return_multi_modal_inputs:
                row_dict["multi_modal_inputs"] = dict(model_inputs)
                row_dict["multi_modal_inputs"].pop("second_per_grid_ts", None)

        else:
            if self.apply_chat_template_kwargs.get("chat_template") is None:
                assert hasattr(self.tokenizer, "chat_template"), (
                    "chat_template should be provided in apply_chat_template_kwargs or tokenizer config, "
                    "models like GLM can copy chat_template.jinja from instruct models"
                )
            apply_kwargs = dict(self.apply_chat_template_kwargs)
            if self.tool_schemas is not None:
                apply_kwargs["tools"] = self.tool_schemas
            raw_prompt = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False, **apply_kwargs
            )
            model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            # qwen-vl mrope
            processor_name = self.processor.__class__.__name__
            tokenizer_path = getattr(self.tokenizer, 'name_or_path', '') or ''

            if "Qwen3VLProcessor" in processor_name:
                from verl.models.transformers.qwen3_vl import get_rope_index
            elif "qwen3" in tokenizer_path.lower() or "Qwen3" in tokenizer_path:
                # Qwen3 model detected but Qwen3VLProcessor not available
                raise RuntimeError(
                    f"Qwen3-VL model detected (tokenizer path: {tokenizer_path}) but Qwen3VLProcessor "
                    f"not found (got {processor_name}). Please upgrade transformers to 4.57+ to properly "
                    f"support Qwen3-VL, or the model may silently use incorrect rope indexing."
                )
            else:
                # Qwen2-VL model
                from verl.models.transformers.qwen2_vl import get_rope_index

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=model_inputs.get("image_grid_thw"),
                video_grid_thw=model_inputs.get("video_grid_thw"),
                second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                attention_mask=attention_mask[0],
            )
            valid_mask = attention_mask[0].bool()
            text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
            text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
            position_ids = [torch.cat((text_position_ids, vision_position_ids), dim=0)]
        elif self.processor is not None and "Glm4vImageProcessor" in self.processor.image_processor.__class__.__name__:
            from verl.models.transformers.glm4v import get_rope_index

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=model_inputs.get("image_grid_thw"),
                video_grid_thw=model_inputs.get("video_grid_thw"),
                attention_mask=attention_mask[0],
            )
            valid_mask = attention_mask[0].bool()
            text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
            text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
            position_ids = [torch.cat((text_position_ids, vision_position_ids), dim=0)]
        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "middle":
                left_half = self.max_prompt_length // 2
                right_half = self.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        row_dict["raw_prompt_ids"] = raw_prompt_ids
        row_dict["raw_prompt"] = messages

        if self.return_full_prompt:
            row_dict["full_prompts"] = raw_prompt

        return self._add_metadata_fields(row_dict, item)

    def _add_metadata_fields(self, row_dict: dict, item: int) -> dict:
        """Add tool training metadata fields to row_dict."""

        # --- Tool training specific fields from rl_dataset_json.py ---

        # Index handling
        row_dict["index"] = row_dict.pop("id", row_dict.get("index", item))

        # Tool training metadata
        title_list = row_dict.pop(self.image_search_title_list_key, None)
        row_dict["image_search_title_list"] = (
            title_list[:self.image_search_max_results] if title_list else None
        )

        thumbnail_list = row_dict.pop(self.image_search_thumbnail_list_key, None)
        row_dict["image_search_thumbnail_list"] = (
            thumbnail_list[:self.image_search_max_results] if thumbnail_list else None
        )

        row_dict["image_search_summary"] = row_dict.pop(self.image_search_summary_key, None)

        # Build tools_kwargs for AgentLoop
        tools_kwargs = row_dict.get("extra_info", {}).get("tools_kwargs", {})
        interaction_kwargs = row_dict.get("extra_info", {}).get("interaction_kwargs", {})
        need_tools_kwargs = row_dict.get("extra_info", {}).get("need_tools_kwargs", self.need_tools_kwargs)

        # Add image_search_tool kwargs (key must match tool name from tool config)
        if row_dict.get("image_search_title_list") or row_dict.get("image_search_thumbnail_list"):
            tools_kwargs["image_search_tool"] = {
                "create_kwargs": {
                    "image_search_title_list": row_dict.get("image_search_title_list"),
                    "image_search_thumbnail_list": row_dict.get("image_search_thumbnail_list"),
                    "image_search_summary": row_dict.get("image_search_summary"),
                }
            }

        if need_tools_kwargs and not tools_kwargs:
            logger.warning("tools_kwargs is empty for index %s, data source: %s", row_dict["index"], row_dict.get("data_source"))

        row_dict["tools_kwargs"] = tools_kwargs
        row_dict["interaction_kwargs"] = interaction_kwargs

        # Build extra_info
        if "extra_info" not in row_dict or row_dict["extra_info"] is None:
            row_dict["extra_info"] = {}
        row_dict["extra_info"]["index"] = row_dict["index"]
        row_dict["extra_info"]["tools_kwargs"] = tools_kwargs
        row_dict["extra_info"]["interaction_kwargs"] = interaction_kwargs

        # Set agent_name for tool training
        row_dict["agent_name"] = "tool_agent"

        # Clean up internal fields
        row_dict.pop("input_template", None)

        return row_dict

    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()
            if "dataframe" in state:
                del state["dataframe"]
            return state
        return self.__dict__.copy()

    def __setstate__(self, state):
        """Restore state from pickle."""
        self.__dict__.update(state)
        # Reload dataframe if it was not serialized (needed for DataLoader workers)
        if "dataframe" not in self.__dict__:
            # in_worker=True: use sequential filtering to avoid nested multiprocessing
            self._read_files_and_tokenize(in_worker=True)
