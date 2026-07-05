import torch
import numpy as np
import cv2


class MaskDirectionalExpansion:
    """Directionally expand or shrink a mask (black & white video mask).

    Positive values expand the mask in that direction; negative values shrink it.
    The 'all' parameter adds to each direction equally.
    When 'use_all_only' is enabled, only the 'all' value is used and individual
    direction inputs are ignored.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "top": (
                    "INT",
                    {"default": 0, "min": -1000, "max": 1000, "step": 1},
                ),
                "bottom": (
                    "INT",
                    {"default": 0, "min": -1000, "max": 1000, "step": 1},
                ),
                "left": (
                    "INT",
                    {"default": 0, "min": -1000, "max": 1000, "step": 1},
                ),
                "right": (
                    "INT",
                    {"default": 0, "min": -1000, "max": 1000, "step": 1},
                ),
                "all": (
                    "INT",
                    {"default": 0, "min": -1000, "max": 1000, "step": 1},
                ),
                "use_all_only": (
                    "BOOLEAN",
                    {"default": False},
                ),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "apply"
    CATEGORY = "mask"

    def apply(
        self,
        mask: torch.Tensor,
        top: int,
        bottom: int,
        left: int,
        right: int,
        all: int,
        use_all_only: bool,
    ):
        if use_all_only:
            top = bottom = left = right = all
        else:
            top += all
            bottom += all
            left += all
            right += all

        # No-op if all values are zero
        if top == 0 and bottom == 0 and left == 0 and right == 0:
            return (mask,)

        processed = []
        for frame in mask:
            arr = (frame.cpu().numpy() * 255.0).astype(np.uint8)
            arr = self._apply_directional(arr, top, bottom, left, right)
            tensor = torch.from_numpy(arr.astype(np.float32) / 255.0)
            processed.append(tensor)

        return (torch.stack(processed, dim=0),)

    @staticmethod
    def _apply_directional(
        arr: np.ndarray, top: int, bottom: int, left: int, right: int
    ) -> np.ndarray:
        """Apply directional dilation (positive) or erosion (negative) to a
        single uint8 mask frame."""

        # --- Top ---
        if top != 0:
            size = abs(top)
            kernel = np.ones((size + 1, 1), np.uint8)
            if top > 0:
                # dilate upward → anchor at the bottom of the kernel
                arr = cv2.dilate(arr, kernel, anchor=(0, size))
            else:
                # erode from top → anchor at the top of the kernel
                arr = cv2.erode(arr, kernel, anchor=(0, 0))

        # --- Bottom ---
        if bottom != 0:
            size = abs(bottom)
            kernel = np.ones((size + 1, 1), np.uint8)
            if bottom > 0:
                # dilate downward → anchor at the top of the kernel
                arr = cv2.dilate(arr, kernel, anchor=(0, 0))
            else:
                # erode from bottom → anchor at the bottom of the kernel
                arr = cv2.erode(arr, kernel, anchor=(0, size))

        # --- Left ---
        if left != 0:
            size = abs(left)
            kernel = np.ones((1, size + 1), np.uint8)
            if left > 0:
                # dilate leftward → anchor at the right of the kernel
                arr = cv2.dilate(arr, kernel, anchor=(size, 0))
            else:
                # erode from left → anchor at the left of the kernel
                arr = cv2.erode(arr, kernel, anchor=(0, 0))

        # --- Right ---
        if right != 0:
            size = abs(right)
            kernel = np.ones((1, size + 1), np.uint8)
            if right > 0:
                # dilate rightward → anchor at the left of the kernel
                arr = cv2.dilate(arr, kernel, anchor=(0, 0))
            else:
                # erode from right → anchor at the right of the kernel
                arr = cv2.erode(arr, kernel, anchor=(size, 0))

        return arr


# ComfyUI registration
NODE_CLASS_MAPPINGS = {
    "MaskDirectionalExpansion": MaskDirectionalExpansion,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MaskDirectionalExpansion": "Mask Directional Expansion",
}
