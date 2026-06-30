from typing import Tuple

import torch


class MaskExpandShrinkDirectional:
    """
    A ComfyUI custom node that expands or shrinks a mask in selected directions.

    Takes a MASK (black and white) as input and outputs a MASK where the white
    masked areas have been expanded (positive amount) or shrunk (negative amount)
    in the selected directions (up, down, left, right) in any combination.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "amount": (
                    "INT",
                    {
                        "default": 0,
                        "min": -9999,
                        "max": 9999,
                        "step": 1,
                        "display": "number",
                    },
                ),
                "direction_up": (["disable", "enable"],),
                "direction_down": (["disable", "enable"],),
                "direction_left": (["disable", "enable"],),
                "direction_right": (["disable", "enable"],),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "process_mask"
    CATEGORY = "mask"

    def process_mask(
        self,
        mask: torch.Tensor,
        amount: int,
        direction_up: str,
        direction_down: str,
        direction_left: str,
        direction_right: str,
    ) -> Tuple[torch.Tensor]:
        # ── Fast path: no change ────────────────────────────────────────────
        if amount == 0:
            return (mask,)

        original_shape = mask.shape
        original_dim = mask.dim()

        # ── Normalise to [B, H, W] ─────────────────────────────────────────
        if original_dim == 2:
            mask = mask.unsqueeze(0)            # [H, W]       -> [1, H, W]
        elif original_dim == 4:
            mask = mask.squeeze(-3)             # [B, 1, H, W] -> [B, H, W]

        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

        # Short aliases
        H = mask.shape[-2]
        W = mask.shape[-1]
        n = amount if amount > 0 else -amount   # abs

        # ── Clamp shift amount to prevent wrap-around ───────────────────────
        if n > H or n > W:
            n = min(n, H, W)

        if n == 0:
            return (mask.view(original_shape),)

        # ── Determine which axes to apply ───────────────────────────────────
        apply_up = direction_up == "enable"
        apply_down = direction_down == "enable"
        apply_left = direction_left == "enable"
        apply_right = direction_right == "enable"

        anything = apply_up | apply_down | apply_left | apply_right
        if not anything:
            return (mask.view(original_shape),)

        # ── Core logic ──────────────────────────────────────────────────────
        # Expansion  (amount > 0): OR  all shifted copies with the original.
        # Shrinkage  (amount < 0): AND all shifted copies with the original.
        #
        # We fold results incrementally via maximum() / minimum() to avoid an
        # extra stacked tensor allocation.

        is_expand = amount > 0

        if is_expand:
            result = mask  # start with original
        else:
            result = mask

        # Helper: create a shifted + zero-filled view (avoids clone+roll).
        # torch.roll creates a new tensor, then we zero the wrapped edge.
        if is_expand:
            # ── EXPAND ──────────────────────────────────────────────────────
            if apply_up:
                shifted = torch.roll(mask, shifts=n, dims=-2)
                if n < H:
                    shifted[..., :n, :] = 0.0
                result = torch.maximum(result, shifted)

            if apply_down:
                shifted = torch.roll(mask, shifts=-n, dims=-2)
                if n < H:
                    shifted[..., -n:, :] = 0.0
                result = torch.maximum(result, shifted)

            if apply_left:
                shifted = torch.roll(mask, shifts=n, dims=-1)
                if n < W:
                    shifted[..., :, :n] = 0.0
                result = torch.maximum(result, shifted)

            if apply_right:
                shifted = torch.roll(mask, shifts=-n, dims=-1)
                if n < W:
                    shifted[..., :, -n:] = 0.0
                result = torch.maximum(result, shifted)

        else:
            # ── SHRINK ──────────────────────────────────────────────────────
            if apply_up:
                shifted = torch.roll(mask, shifts=-n, dims=-2)
                if n < H:
                    shifted[..., -n:, :] = 0.0
                result = torch.minimum(result, shifted)

            if apply_down:
                shifted = torch.roll(mask, shifts=n, dims=-2)
                if n < H:
                    shifted[..., :n, :] = 0.0
                result = torch.minimum(result, shifted)

            if apply_left:
                shifted = torch.roll(mask, shifts=-n, dims=-1)
                if n < W:
                    shifted[..., :, -n:] = 0.0
                result = torch.minimum(result, shifted)

            if apply_right:
                shifted = torch.roll(mask, shifts=n, dims=-1)
                if n < W:
                    shifted[..., :, :n] = 0.0
                result = torch.minimum(result, shifted)

        # ── Restore original dimensionality ─────────────────────────────────
        if original_dim == 2:
            result = result.squeeze(0)
        elif original_dim == 4:
            result = result.unsqueeze(-3)

        return (result,)


# ── ComfyUI registration ──────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "MaskExpandShrinkDirectional": MaskExpandShrinkDirectional,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MaskExpandShrinkDirectional": "Mask Expand / Shrink (Directional)",
}
