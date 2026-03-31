#!/usr/bin/env python3
"""Inspect dataset roots and discover RGB frame availability."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Discover RGB frames in a dataset root.")
    parser.add_argument("--dataset", choices=["scannet", "scannetpp", "arkitscenes"], required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--max-examples", type=int, default=10)
    return parser.parse_args()


def list_image_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def detect_scannet_scenes(root: Path) -> tuple[list[Path], list[str], list[str]]:
    scenes: list[Path] = []
    missing: list[str] = []
    ambiguous: list[str] = []

    for scene_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        color = scene_dir / "color"
        sens = list(scene_dir.glob("*.sens"))
        if color.exists():
            scenes.append(scene_dir)
        elif sens:
            missing.append(f"{scene_dir.name}: only .sens found ({len(sens)} file(s)); expected extracted color/ frames")
        else:
            ambiguous.append(f"{scene_dir.name}: no color/ folder and no .sens file")
    return scenes, missing, ambiguous


def detect_scannetpp_scenes(root: Path) -> tuple[list[Path], list[str], list[str]]:
    scenes: list[Path] = []
    missing: list[str] = []
    ambiguous: list[str] = []

    for scene_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        target = scene_dir / "dslr" / "resized_undistorted_images"
        if target.exists():
            scenes.append(scene_dir)
        elif (scene_dir / "dslr").exists():
            missing.append(f"{scene_dir.name}: dslr exists but resized_undistorted_images missing")
        else:
            ambiguous.append(f"{scene_dir.name}: expected dslr/resized_undistorted_images")
    return scenes, missing, ambiguous


def detect_arkitscenes_scenes(root: Path) -> tuple[list[Path], list[str], list[str]]:
    scenes: list[Path] = []
    missing: list[str] = []
    ambiguous: list[str] = []

    for scene_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        lowres = scene_dir / "lowres_wide"
        vga = scene_dir / "vga_wide"
        if lowres.exists() or vga.exists():
            scenes.append(scene_dir)
        else:
            ambiguous.append(f"{scene_dir.name}: expected lowres_wide/ or vga_wide/")
    return scenes, missing, ambiguous


def scene_image_roots(dataset: str, scene: Path) -> Iterable[Path]:
    if dataset == "scannet":
        yield scene / "color"
    elif dataset == "scannetpp":
        yield scene / "dslr" / "resized_undistorted_images"
    else:
        for name in ("lowres_wide", "vga_wide"):
            path = scene / name
            if path.exists():
                yield path


def main() -> None:
    args = parse_args()
    root = args.input_root
    if not root.exists():
        raise FileNotFoundError(f"Input root does not exist: {root}")

    detector = {
        "scannet": detect_scannet_scenes,
        "scannetpp": detect_scannetpp_scenes,
        "arkitscenes": detect_arkitscenes_scenes,
    }[args.dataset]

    scenes, missing, ambiguous = detector(root)

    scene_image_counts: dict[str, int] = {}
    all_examples: list[str] = []
    total_images = 0

    for scene in scenes:
        count = 0
        for img_root in scene_image_roots(args.dataset, scene):
            imgs = list_image_files(img_root)
            count += len(imgs)
            if len(all_examples) < args.max_examples:
                all_examples.extend([str(p) for p in imgs[: max(0, args.max_examples - len(all_examples))]])
        scene_image_counts[scene.name] = count
        total_images += count

    report = {
        "dataset": args.dataset,
        "input_root": str(root.resolve()),
        "num_scenes": len(scenes),
        "num_discovered_rgb_images": total_images,
        "example_paths": all_examples[: args.max_examples],
        "missing_or_warnings": missing,
        "ambiguous_folders": ambiguous,
        "scene_image_counts": scene_image_counts,
    }

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
