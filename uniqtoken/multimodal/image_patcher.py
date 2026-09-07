from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class ImagePatch:
    """
    Represents an extracted 2D image patch with 2D spatial coordinate metadata.
    """

    patch_id: int
    row: int
    col: int
    grid_h: int
    grid_w: int
    norm_bbox: Tuple[float, float, float, float]  # (y1, x1, y2, x2) in [0.0, 1.0]
    pixels: List[float]  # Flattened normalized pixel values (P * P * C)
    scale: float = 1.0  # Normalization scale used at extraction: raw = norm * scale + offset
    offset: float = 0.0  # Normalization offset used at extraction


class DynamicImagePatcher:
    """
    Dynamic Aspect-Ratio Image Patching & 2D Spatial Coordinate Engine.

    Slices arbitrary 2D/3D images (H x W x C) into uniform P x P pixel patches
    while preserving exact spatial aspect ratios and 2D grid coordinates.
    """

    def __init__(
        self,
        patch_size: int = 16,
        channels: int = 3,
        normalize_pixels: bool = True,
        pixel_range: Optional[Tuple[float, float]] = None,  # (min, max) if known, e.g., (0.0, 1.0) or (0.0, 255.0)
    ):
        if patch_size <= 0:
            raise ValueError("patch_size must be positive")
        if channels <= 0:
            raise ValueError("channels must be positive")
        if pixel_range is not None:
            if len(pixel_range) != 2 or pixel_range[0] >= pixel_range[1]:
                raise ValueError("pixel_range must be (min, max) with min < max")

        self.patch_size = patch_size
        self.channels = channels
        self.normalize_pixels = normalize_pixels
        self.pixel_range = pixel_range
        self.patch_dim = patch_size * patch_size * channels

    def extract_patches(
        self,
        image_pixels: List[List[List[float]]],  # Shape: [H][W][C]
    ) -> Tuple[List[ImagePatch], Tuple[int, int]]:
        """
        Extracts non-overlapping patches from a 3D pixel array [H][W][C].
        Returns (list_of_patches, (grid_h, grid_w)).

        Each patch is stamped with the (scale, offset) normalization actually
        applied, so `reconstruct_image` can invert it exactly.
        """
        if not image_pixels:
            return [], (0, 0)

        h = len(image_pixels)
        w = len(image_pixels[0]) if h > 0 else 0
        if w == 0:
            # An empty first row must not mask non-empty later rows; the
            # jagged-width validation below will then reject the image.
            w = max((len(row) for row in image_pixels), default=0)

        if h == 0 or w == 0:
            return [], (0, 0)

        # Validate image dimensions
        for row in image_pixels:
            if len(row) != w:
                raise ValueError("All rows must have the same width")
            for pixel in row:
                if len(pixel) != self.channels:
                    raise ValueError(f"Expected {self.channels} channels, got {len(pixel)}")

        p = self.patch_size
        grid_h = math.ceil(h / p)
        grid_w = math.ceil(w / p)

        # Determine normalization scale
        if self.pixel_range is not None:
            # Explicit range provided
            scale = self.pixel_range[1] - self.pixel_range[0]
            offset = self.pixel_range[0]
        elif self.normalize_pixels:
            # Auto-detect: check if any pixel exceeds 1.0 significantly
            # Use a threshold to avoid false positives from floating-point noise
            vals = [val for row in image_pixels for px in row for val in px]
            max_val = max(vals) if vals else 1.0
            if max_val > 1.0 + 1e-6:
                # Treat as [0, 255] or similar range
                scale = 255.0
                offset = 0.0
            else:
                # Already in [0, 1] range
                scale = 1.0
                offset = 0.0
        else:
            scale = 1.0
            offset = 0.0

        patches: List[ImagePatch] = []
        patch_idx = 0

        for gh in range(grid_h):
            for gw in range(grid_w):
                row_start = gh * p
                col_start = gw * p
                row_end = min(row_start + p, h)
                col_end = min(col_start + p, w)

                # Extract and pad patch if boundary is uneven
                patch_data: List[float] = []
                for r in range(row_start, row_start + p):
                    for c in range(col_start, col_start + p):
                        if r < h and c < w:
                            pixel_vals = image_pixels[r][c]
                        else:
                            # Use the raw lower bound so padding is zero after normalization.
                            pixel_vals = [offset] * self.channels

                        for val in pixel_vals:
                            # Normalize: (val - offset) / scale
                            patch_data.append((val - offset) / scale)

                # Compute normalized bounding box (y1, x1, y2, x2) in [0.0, 1.0]
                y1 = row_start / h
                x1 = col_start / w
                y2 = row_end / h
                x2 = col_end / w

                patches.append(
                    ImagePatch(
                        patch_id=patch_idx,
                        row=gh,
                        col=gw,
                        grid_h=grid_h,
                        grid_w=grid_w,
                        norm_bbox=(
                            round(y1, 4),
                            round(x1, 4),
                            round(y2, 4),
                            round(x2, 4),
                        ),
                        pixels=patch_data,
                        scale=scale,
                        offset=offset,
                    )
                )
                patch_idx += 1

        return patches, (grid_h, grid_w)

    def reconstruct_image(
        self,
        patches: List[ImagePatch],
        grid_h: int,
        grid_w: int,
        *,
        scale: Optional[float] = None,
        offset: Optional[float] = None,
    ) -> List[List[List[float]]]:
        """
        Reconstructs the 2D image matrix [H][W][C] from a sequence of ImagePatch instances.

        Normalization is inverted per patch (`raw = norm * scale + offset`):
        explicit keyword arguments win, otherwise each patch's own stamped
        values are used, so an extract/reconstruct roundtrip restores the
        original pixel range exactly.
        """
        if not patches:
            return []

        p = self.patch_size
        h = grid_h * p
        w = grid_w * p
        c = self.channels

        # Initialize canvas
        canvas = [[[0.0] * c for _ in range(w)] for _ in range(h)]

        for patch in patches:
            if not 0 <= patch.row < grid_h or not 0 <= patch.col < grid_w:
                continue  # Skip invalid patches (also guards negative indices)

            row_start = patch.row * p
            col_start = patch.col * p
            pix_idx = 0
            patch_scale = scale if scale is not None else patch.scale
            patch_offset = offset if offset is not None else patch.offset

            for r in range(row_start, min(row_start + p, h)):
                for col in range(col_start, min(col_start + p, w)):
                    for ch in range(c):
                        if pix_idx < len(patch.pixels):
                            canvas[r][col][ch] = patch.pixels[pix_idx] * patch_scale + patch_offset
                            pix_idx += 1

        return canvas
