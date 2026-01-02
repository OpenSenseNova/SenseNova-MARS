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
"""Image processing utilities for VLM tools."""

import base64
import io
import math
import os
import random
from io import BytesIO
from typing import Optional, Union

from PIL import Image

try:
    from qwen_vl_utils import smart_resize
except ImportError:
    smart_resize = None


# Default pixel limits aligned with env var settings
# MIN_PIXELS = 65536 (256x256), MAX_PIXELS = 8294400 (~4K resolution)
DEFAULT_MIN_PIXELS = 65536
DEFAULT_MAX_PIXELS = 8294400

# Default factor for Qwen3-VL (patch_size=16, merge_size=2 -> factor=32)
# Qwen2-VL uses factor=28 (patch_size=14, merge_size=2)
DEFAULT_FACTOR = 32


def get_default_pixel_limits() -> tuple[int, int]:
    """
    Return (min_pixels, max_pixels) defaults used for image preprocessing.

    Defaults:
      MIN_PIXELS = 65536 (256x256), MAX_PIXELS = 8294400 (~4K resolution).
    """
    min_pixels = int(os.getenv("MIN_PIXELS", DEFAULT_MIN_PIXELS))
    max_pixels = int(os.getenv("MAX_PIXELS", DEFAULT_MAX_PIXELS))
    return min_pixels, max_pixels


def get_default_factor() -> int:
    """
    Return the default factor for image resizing.

    Factor = patch_size * merge_size
    - Qwen3-VL: 16 * 2 = 32
    - Qwen2-VL: 14 * 2 = 28
    """
    return int(os.getenv("IMAGE_FACTOR", DEFAULT_FACTOR))


def process_image(
    image: Union[dict, Image.Image],
    max_pixels: Optional[int] = None,
    min_pixels: Optional[int] = None,
    max_aspect_ratio: float = 100,
    use_smart_resize: bool = True,
    factor: Optional[int] = None,
) -> Image.Image:
    """
    Process image with optional smart_resize.
    Aligned with dataset processing logic for consistency.

    Args:
        image: Image dict with 'bytes' or PIL Image
        max_pixels: Maximum pixels (from env var if None)
        min_pixels: Minimum pixels (from env var if None)
        max_aspect_ratio: Max aspect ratio
        use_smart_resize: If True, use smart_resize (default: True)
        factor: Alignment factor (default: 32 for Qwen3-VL, 28 for Qwen2-VL)

    Returns:
        Processed PIL Image
    """
    if isinstance(image, dict):
        image = Image.open(BytesIO(image["bytes"]))

    # Convert to RGB before resize (consistent with dataset processing)
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Get min/max pixels from environment if not provided
    if max_pixels is None or min_pixels is None:
        default_min, default_max = get_default_pixel_limits()
        if max_pixels is None:
            max_pixels = default_max
        if min_pixels is None:
            min_pixels = default_min

    # Get factor from environment if not provided
    if factor is None:
        factor = get_default_factor()

    width, height = image.width, image.height

    # Handle aspect ratio
    current_ratio = max(width, height) / min(width, height) if min(width, height) > 0 else 1
    if current_ratio > max_aspect_ratio:
        if width > height:
            width = int(height * max_aspect_ratio)
        else:
            height = int(width * max_aspect_ratio)

    # Use smart_resize if requested (aligned with dataset processing)
    if use_smart_resize and smart_resize is not None:
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        # Use BICUBIC resampling to match HF processor
        image = image.resize((resized_width, resized_height), Image.BICUBIC)
    else:
        # Legacy mode (old logic)
        resize_factor = 1

        if (width * height) > max_pixels:
            resize_factor = math.sqrt(max_pixels / (width * height))
        elif (width * height) < min_pixels:
            resize_factor = math.sqrt(min_pixels / (width * height))

        short_side = min(width, height)
        if short_side < factor:
            resize_factor = max(resize_factor, factor / short_side)

        width, height = int(width * resize_factor), int(height * resize_factor)
        image = image.resize((width, height), Image.BICUBIC)

    return image


def base64_to_pil(image_base64: str) -> Image.Image:
    """Convert base64 string to PIL Image."""
    image_data = base64.b64decode(image_base64)
    image_io = io.BytesIO(image_data)
    image = Image.open(image_io)
    return image


def pil_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """Convert PIL Image to base64 string."""
    buffered = io.BytesIO()
    image.save(buffered, format=format)
    image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return image_base64


def resize_cropped_image(
    image: Image.Image,
    min_pixels: Optional[int] = None,
    max_pixels: Optional[int] = None,
    factor: Optional[int] = None,
) -> Image.Image:
    """
    Resize cropped image using smart_resize (same as Pixel-Reasoner).
    This ensures the image dimensions are properly aligned with the model's factor.

    Args:
        image: PIL Image to resize
        min_pixels: Minimum pixels (from env var if None)
        max_pixels: Maximum pixels (from env var if None)
        factor: Alignment factor (default: 32 for Qwen3-VL, 28 for Qwen2-VL)

    Returns:
        Resized PIL Image
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    original_width, original_height = image.size
    width, height = original_width, original_height

    # Get min/max pixels from environment if not provided
    if max_pixels is None or min_pixels is None:
        default_min, default_max = get_default_pixel_limits()
        if max_pixels is None:
            max_pixels = default_max
        if min_pixels is None:
            min_pixels = default_min

    # Get factor from environment if not provided
    if factor is None:
        factor = get_default_factor()

    # Sample 1/64 (~1.5%) of logs to reduce spam while maintaining observability
    do_print = random.randint(1, 64) == 1

    if smart_resize is not None:
        # Use smart_resize from qwen_vl_utils (same as Pixel-Reasoner)
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        # Use BICUBIC resampling to match HF processor
        image = image.resize((resized_width, resized_height), Image.BICUBIC)
        if do_print:
            print(
                f"[image_utils] resize_cropped_image: {original_width}x{original_height} -> "
                f"{resized_width}x{resized_height} (smart_resize, pixels={resized_width * resized_height}, "
                f"min_pixels={min_pixels}, max_pixels={max_pixels}, factor={factor})",
                flush=True,
            )
    else:
        # Fallback without smart_resize
        resize_factor = 1
        if (width * height) > max_pixels:
            resize_factor = math.sqrt(max_pixels / (width * height))
        elif (width * height) < min_pixels:
            resize_factor = math.sqrt(min_pixels / (width * height))

        width, height = int(width * resize_factor), int(height * resize_factor)
        # Align to factor
        width = (width // factor) * factor
        height = (height // factor) * factor
        width = max(width, factor)
        height = max(height, factor)
        image = image.resize((width, height), Image.BICUBIC)
        if do_print:
            print(
                f"[image_utils] resize_cropped_image: {original_width}x{original_height} -> "
                f"{width}x{height} (fallback, pixels={width * height}, "
                f"min_pixels={min_pixels}, max_pixels={max_pixels}, factor={factor})",
                flush=True,
            )

    return image


def find_image_in_history(
    img_idx: int,
    images: list,
) -> Optional[Image.Image]:
    """
    Find the image at the specified index in the image list.

    Args:
        img_idx: 0-based index of the image (0 = first image)
        images: List of images (PIL Image objects or dicts)

    Returns:
        PIL Image if found, None otherwise
    """
    if images is None or len(images) == 0:
        return None

    if img_idx < 0 or img_idx >= len(images):
        return None

    image = images[img_idx]

    # Handle dict format
    if isinstance(image, dict):
        if "bytes" in image:
            image = Image.open(BytesIO(image["bytes"]))
        elif "image" in image:
            image = image["image"]

    return image
