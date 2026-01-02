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
"""
Image Search Tool for VLM multi-turn training.

This is a DATA-DRIVEN tool - results come from the dataset, not from a live API.
The model calls image_search_tool() with no arguments, and the tool returns
pre-computed reverse image search results from the dataset.

Data format expected in tools_kwargs:
{
    "image_search_tool": {
        "create_kwargs": {
            "image_search_title_list": ["title1", "title2", ...],
            "image_search_thumbnail_list": ["path/to/thumb1.jpg", ...],
            "image_search_summary": "Optional summary text"
        }
    }
}
"""

import logging
import os
from typing import Any, Optional
from uuid import uuid4

from PIL import Image

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema, ToolResponse
from .utils.image_utils import process_image

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class ImageSearchTool(BaseTool):
    """
    A tool for reverse image search results.

    This is a DATA-DRIVEN tool - the search results are pre-computed and
    stored in the dataset. When the model calls this tool, it returns
    the pre-computed results including titles and thumbnail images.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """
        Initialize ImageSearchTool.

        Config options:
            use_image_search_summary: Include LLM summary in results (default: False)
            min_pixels: Min pixels for thumbnail resize
            max_pixels: Max pixels for thumbnail resize
        """
        super().__init__(config, tool_schema)
        self._instance_dict = {}

        # Config
        self.use_image_search_summary = config.get("use_image_search_summary", False)
        self.min_pixels = config.get("min_pixels", None)
        self.max_pixels = config.get("max_pixels", None)

        logger.info(f"Initialized ImageSearchTool: use_summary={self.use_image_search_summary}")

    def _load_image(self, path: str) -> Optional[Image.Image]:
        """Load image from local file path."""
        try:
            if os.path.exists(path):
                return Image.open(path)
            else:
                logger.warning(f"Image not found: {path}")
                return None
        except Exception as e:
            logger.warning(f"Failed to load image {path}: {e}")
            return None

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        """
        Create a tool instance with pre-computed search results.

        Args:
            instance_id: Optional unique identifier
            **kwargs: Should contain create_kwargs with:
                - image_search_title_list: List of result titles
                - image_search_thumbnail_list: List of thumbnail paths
                - image_search_summary: Optional summary text

        Returns:
            Tuple of (instance_id, ToolResponse)
        """
        if instance_id is None:
            instance_id = str(uuid4())

        # Handle create_kwargs parameter if passed
        create_kwargs = kwargs.get("create_kwargs", {})
        if create_kwargs:
            kwargs.update(create_kwargs)

        # Extract search results from kwargs
        title_list = kwargs.get("image_search_title_list", [])
        thumbnail_list = kwargs.get("image_search_thumbnail_list", [])
        summary = kwargs.get("image_search_summary", None)

        self._instance_dict[instance_id] = {
            "title_list": title_list,
            "thumbnail_list": thumbnail_list,
            "summary": summary,
            "called": False,
        }

        return instance_id, ToolResponse()

    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs
    ) -> tuple[ToolResponse, float, dict]:
        """
        Execute image search - returns pre-computed results.

        Args:
            instance_id: Instance ID from create()
            parameters: Tool parameters (typically empty for image_search)
            **kwargs: Additional kwargs

        Returns:
            Tuple of (ToolResponse with text and images, reward, info_dict)
        """
        instance_data = self._instance_dict.get(instance_id, {})
        title_list = instance_data.get("title_list", [])
        thumbnail_list = instance_data.get("thumbnail_list", [])
        summary = instance_data.get("summary", None)

        # Mark as called
        if instance_id in self._instance_dict:
            self._instance_dict[instance_id]["called"] = True

        # Handle case with no results
        # Note: <tool_response> tags are added by the chat template, not here
        if not title_list or not thumbnail_list:
            return (
                ToolResponse(text="No matching images were found."),
                0.0,
                {"success": True, "num_results": 0},
            )

        # Load and process thumbnails
        thumbnail_images = []
        for path in thumbnail_list:
            img = self._load_image(path)
            if img is not None:
                # Process image with consistent sizing
                img = process_image(
                    img,
                    min_pixels=self.min_pixels,
                    max_pixels=self.max_pixels,
                )
                thumbnail_images.append(img)

        # Build interleaved content
        # Interleaves: text -> image -> text -> image -> ... -> text
        # Note: <tool_response> tags are added by the chat template, not here
        interleaved_content = []

        # Header
        interleaved_content.append({"type": "text", "text": "Reverse Image Search Results:"})

        # Interleave titles and thumbnails
        for i, title in enumerate(title_list):
            # Add title and thumbnail label
            interleaved_content.append({"type": "text", "text": f"\n\nTitle {i + 1}: {title}\nThumbnail {i + 1}: "})
            # Add thumbnail image if available
            if i < len(thumbnail_images):
                interleaved_content.append({"type": "image", "image": thumbnail_images[i]})

        # Add summary if enabled and available
        if self.use_image_search_summary and summary:
            interleaved_content.append({"type": "text", "text": f"\n\nSummary of search results: {summary}"})


        return (
            ToolResponse(
                interleaved_content=interleaved_content,
            ),
            0.0,
            {"success": True, "num_results": len(title_list)},
        )

    async def release(self, instance_id: str, **kwargs) -> None:
        """Release the tool instance."""
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
