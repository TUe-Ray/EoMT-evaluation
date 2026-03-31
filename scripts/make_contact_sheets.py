#!/usr/bin/env python3
"""Rebuild contact sheets from existing per-frame outputs."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create contact sheets from existing output artifacts.")
    parser.add_argument("--outputs-root", type=Path, required=True)
    parser.add_argument("--max-masks", type=int, default=8)
    return parser.parse_args()


def build_sheet(frame_dir: Path, max_masks: int) -> None:
    original = frame_dir / "original.png"
    overlay = frame_dir / "overlay.png"
    masks_file = frame_dir / "masks.npz"
    if not (original.exists() and overlay.exists() and masks_file.exists()):
        return

    image = np.array(Image.open(original).convert("RGB"))
    over = np.array(Image.open(overlay).convert("RGB"))
    masks = np.load(masks_file)["masks"]

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    axes[0].imshow(image)
    axes[0].set_title("original")
    axes[0].axis("off")
    axes[1].imshow(over)
    axes[1].set_title("overlay")
    axes[1].axis("off")

    for i in range(8):
        ax = axes[i + 2]
        if i < min(max_masks, masks.shape[0]):
            ax.imshow(masks[i], cmap="gray")
            ax.set_title(f"mask_{i}")
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(frame_dir / "contact_sheet.png")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    for frame_dir in args.outputs_root.rglob("*"):
        if frame_dir.is_dir():
            build_sheet(frame_dir, args.max_masks)


if __name__ == "__main__":
    main()
