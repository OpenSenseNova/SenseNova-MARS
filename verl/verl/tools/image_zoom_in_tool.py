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
Image Zoom In Tool for VLM multi-turn training.

Supports:
- Relative coordinates (0-1000 for Qwen3-VL) or absolute pixel coordinates (Qwen2.5-VL)
- Absolute pixel coordinates
- Padding around crop region (capped at 600px)
- img_idx parameter (0-based) to select which image to zoom
- label parameter for object identification
- smart_resize for output alignment
"""

import ast
import logging
import os
from typing import Any, Optional
from uuid import uuid4

from PIL import Image

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema, ToolResponse
from .utils.image_utils import find_image_in_history, resize_cropped_image

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class ImageZoomInTool(BaseTool):
    """
    A tool for zooming into images with support for:
    - Relative coordinates (Qwen3-VL: 0-1000) or absolute pixel coordinates (Qwen2.5-VL)
    - Absolute pixel coordinates
    - Padding around crop region
    - img_idx (0-based) to select which image from conversation
    - label for object identification
    - smart_resize for output
    """

    MIN_DIMENSION = 28

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """
        Initialize ImageZoomInTool.

        Config options:
            image_crop_mode: "relative" or "absolute" (default: "relative")
            relative_coord_max: Max value for relative coords (1000 for Qwen3-VL)
            padding: [pad_x, pad_y] padding as fraction of image size (default: [0.2, 0.2])
            use_smart_resize: Whether to use smart_resize on output (default: True)
            min_pixels: Min pixels for resize (default: from env or 65536)
            max_pixels: Max pixels for resize (default: from env or 3686400)
        """
        super().__init__(config, tool_schema)
        self._instance_dict = {}

        # Coordinate mode configuration - support both old (crop_image_mode) and new (image_crop_mode) config keys
        self.image_crop_mode = config.get("image_crop_mode") or config.get("crop_image_mode", "relative")
        self.relative_coord_max = float(config.get("relative_coord_max", 1000.0))

        # Padding configuration
        padding = config.get("padding", [0.2, 0.2])
        if isinstance(padding, (list, tuple)) and len(padding) == 2:
            self.padding = (float(padding[0]), float(padding[1]))
        else:
            self.padding = (0.2, 0.2)

        # Resize configuration
        self.use_smart_resize = config.get("use_smart_resize", True)
        self.min_pixels = config.get("min_pixels", None)
        self.max_pixels = config.get("max_pixels", None)

        logger.info(
            f"Initialized ImageZoomInTool: mode={self.image_crop_mode}, "
            f"relative_coord_max={self.relative_coord_max}, padding={self.padding}"
        )

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        """
        Create a tool instance.

        Args:
            instance_id: Optional unique identifier
            **kwargs: Should contain 'images' key with list of images available for zooming
                     (from conversation history)

        Returns:
            Tuple of (instance_id, ToolResponse)
        """
        if instance_id is None:
            instance_id = str(uuid4())

        # Handle create_kwargs parameter if passed
        create_kwargs = kwargs.get("create_kwargs", {})
        if create_kwargs:
            kwargs.update(create_kwargs)

        # Get images from kwargs (list of images from conversation)
        images = kwargs.get("images", [])

        self._instance_dict[instance_id] = {
            "images": images,
            "response": "",
            "reward": 0.0,
        }
        return instance_id, ToolResponse()

    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs
    ) -> tuple[ToolResponse, float, dict]:
        """
        Execute the zoom-in operation.

        Args:
            instance_id: Instance ID from create()
            parameters: Tool parameters including:
                - bbox or bbox_2d: [x1, y1, x2, y2] bounding box
                - img_idx or image_index: 0-based index of image to zoom (default: 0)
                - label: Name/label of the object being zoomed (optional)
            **kwargs: Additional kwargs, may include 'images' to override instance images

        Returns:
            Tuple of (ToolResponse, reward, info_dict)
        """
        # Accept both 'bbox' (new) and 'bbox_2d' (legacy) parameter names
        bbox = parameters.get("bbox") or parameters.get("bbox_2d")
        # Accept both 'img_idx' and 'image_index', both 0-based
        img_idx = parameters.get("img_idx") if parameters.get("img_idx") is not None else parameters.get("image_index", 0)
        label = parameters.get("label", "")

        # Validate bbox
        if not bbox or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return (
                ToolResponse(text="Error: bbox parameter is missing or not a list of 4 numbers."),
                -0.05,
                {"success": False, "error": "invalid_bbox"},
            )

        # Get images - prefer kwargs override, then instance dict
        images = kwargs.get("images")
        if images is None:
            instance_data = self._instance_dict.get(instance_id, {})
            images = instance_data.get("images", [])

        # Find the target image
        if not images:
            return (
                ToolResponse(text="Error: No images available for zooming."),
                -0.05,
                {"success": False, "error": "no_images"},
            )

        image = find_image_in_history(img_idx, images)
        if image is None:
            return (
                ToolResponse(
                    text=f"Error: Image at index {img_idx} not found. "
                    f"Available images: {len(images) if images else 0}"
                ),
                -0.05,
                {"success": False, "error": "image_not_found"},
            )

        # Ensure we have a PIL Image
        if not isinstance(image, Image.Image):
            try:
                from qwen_vl_utils import fetch_image

                image = fetch_image({"image": image})
            except Exception as e:
                return (
                    ToolResponse(text=f"Error: Failed to load image: {e}"),
                    -0.05,
                    {"success": False, "error": "image_load_failed"},
                )

        # Perform the crop
        success, result = self._crop_image(image, bbox)

        if not success:
            return (
                ToolResponse(text=f"Error: {result}"),
                -0.05,
                {"success": False, "error": "crop_failed"},
            )

        cropped_image = result

        # Apply smart_resize if enabled
        if self.use_smart_resize:
            cropped_image = resize_cropped_image(
                cropped_image,
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels,
            )

        # Note: <tool_response> tags are added by the chat template, not here
        return (
            ToolResponse(
                image=[cropped_image],
                text="Here is the zoomed image:",
            ),
            0.0,
            {"success": True},
        )

    def _crop_image(
        self, image: Image.Image, bbox: list
    ) -> tuple[bool, Any]:
        """
        Crop the image based on bounding box coordinates.

        Args:
            image: PIL Image to crop
            bbox: [x1, y1, x2, y2] coordinates

        Returns:
            (success, result) where result is cropped image or error message
        """
        # Parse bbox if string
        if isinstance(bbox, str):
            try:
                bbox = ast.literal_eval(bbox)
            except (ValueError, SyntaxError):
                return False, f"Invalid bbox format: {bbox}"

        if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
            return False, f"Invalid bbox format: {bbox}"

        img_w, img_h = image.size

        # Calculate padding cap at 600px
        padding_cap = (600.0 / img_w, 600.0 / img_h)
        actual_padding = (
            min(self.padding[0], padding_cap[0]),
            min(self.padding[1], padding_cap[1]),
        )

        try:
            x1, y1, x2, y2 = [float(coord) for coord in bbox]
        except (ValueError, TypeError):
            return False, f"Bbox coordinates must be numeric: {bbox}"

        # Handle different coordinate modes
        if self.image_crop_mode == "relative":
            # Validate relative coordinates are within expected range
            if not (
                0 <= x1 <= self.relative_coord_max
                and 0 <= y1 <= self.relative_coord_max
                and 0 <= x2 <= self.relative_coord_max
                and 0 <= y2 <= self.relative_coord_max
            ):
                return (
                    False,
                    f"In relative mode, coordinates must be in [0, {self.relative_coord_max}]. Got: {bbox}",
                )

            # Normalize to 0-1 range
            x1 = x1 / self.relative_coord_max
            y1 = y1 / self.relative_coord_max
            x2 = x2 / self.relative_coord_max
            y2 = y2 / self.relative_coord_max

        elif self.image_crop_mode == "absolute":
            # Convert absolute pixel coordinates to normalized 0-1
            if not (0 <= x1 <= img_w and 0 <= x2 <= img_w and 0 <= y1 <= img_h and 0 <= y2 <= img_h):
                return (
                    False,
                    f"In absolute mode, coordinates must be within image bounds "
                    f"[0, {img_w}] x [0, {img_h}]. Got: {bbox}",
                )

            x1, y1, x2, y2 = x1 / img_w, y1 / img_h, x2 / img_w, y2 / img_h
        else:
            return False, f"Invalid image_crop_mode: {self.image_crop_mode}"

        # Validate ordering
        if not (x1 < x2 and y1 < y2):
            return (
                False,
                f"Invalid bbox coordinates: x1 must be < x2, y1 must be < y2. Got: {bbox}",
            )

        # Apply padding (in normalized coords)
        x1 = max(0.0, x1 - actual_padding[0])
        y1 = max(0.0, y1 - actual_padding[1])
        x2 = min(1.0, x2 + actual_padding[0])
        y2 = min(1.0, y2 + actual_padding[1])

        # Convert back to pixel coordinates and crop
        crop_box = (
            int(x1 * img_w),
            int(y1 * img_h),
            int(x2 * img_w),
            int(y2 * img_h),
        )
        cropped_img = image.crop(crop_box)

        # Resize if too small
        w, h = cropped_img.size
        if w < self.MIN_DIMENSION or h < self.MIN_DIMENSION:
            cropped_img = cropped_img.resize(
                (max(w, self.MIN_DIMENSION), max(h, self.MIN_DIMENSION)),
                Image.Resampling.LANCZOS,
            )

        return True, cropped_img

    async def release(self, instance_id: str, **kwargs) -> None:
        """Release the tool instance."""
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
