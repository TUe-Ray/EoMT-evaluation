#!/usr/bin/env python3
"""Build tiny dataset subsets for EoMT evaluation."""
from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build small, reproducible subsets from indoor datasets.")
    parser.add_argument("--dataset", choices=["scannet", "scannetpp", "arkitscenes"], required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--num-scenes", type=int, default=2)
    parser.add_argument("--frames-per-scene", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--copy-mode", choices=["copy", "symlink", "manifest"], default="manifest")
    parser.add_argument("--prefer-pattern", type=str, default=None)
    return parser.parse_args()


def discover_scenes(dataset: str, input_root: Path) -> list[tuple[str, list[Path]]]:
    scenes: list[tuple[str, list[Path]]] = []
    for scene_dir in sorted(p for p in input_root.iterdir() if p.is_dir()):
        roots: list[Path] = []
        if dataset == "scannet":
            color = scene_dir / "color"
            if color.exists():
                roots = [color]
            elif list(scene_dir.glob("*.sens")):
                print(f"[warn] {scene_dir.name}: found .sens but no extracted color/ frames; skipping")
                continue
        elif dataset == "scannetpp":
            target = scene_dir / "dslr" / "resized_undistorted_images"
            if target.exists():
                roots = [target]
        elif dataset == "arkitscenes":
            if (scene_dir / "lowres_wide").exists():
                roots = [scene_dir / "lowres_wide"]
            elif (scene_dir / "vga_wide").exists():
                roots = [scene_dir / "vga_wide"]
        if roots:
            scenes.append((scene_dir.name, roots))
    return scenes


def collect_images(roots: list[Path], prefer_pattern: str | None = None) -> list[Path]:
    images: list[Path] = []
    for root in roots:
        images.extend([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS])
    images = sorted(images)
    if prefer_pattern:
        preferred = [p for p in images if prefer_pattern in str(p)]
        if preferred:
            return preferred
    return images


def sample_evenly(items: list[Path], k: int) -> list[Path]:
    if not items:
        return []
    if len(items) <= k:
        return items
    idxs = [round(i * (len(items) - 1) / (k - 1)) for i in range(k)]
    return [items[i] for i in idxs]


def safe_link_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        dst.symlink_to(src.resolve())


def main() -> None:
    args = parse_args()
    if not args.input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {args.input_root}")

    random.seed(args.seed)
    args.output_root.mkdir(parents=True, exist_ok=True)

    scenes = discover_scenes(args.dataset, args.input_root)
    if not scenes:
        raise RuntimeError(f"No usable scenes found for dataset={args.dataset} at {args.input_root}")

    random.shuffle(scenes)
    selected_scenes = scenes[: min(args.num_scenes, len(scenes))]

    rows: list[dict[str, str]] = []
    summary = {
        "dataset": args.dataset,
        "input_root": str(args.input_root.resolve()),
        "output_root": str(args.output_root.resolve()),
        "num_scenes_requested": args.num_scenes,
        "num_scenes_selected": len(selected_scenes),
        "frames_per_scene": args.frames_per_scene,
        "copy_mode": args.copy_mode,
        "seed": args.seed,
        "scenes": {},
    }

    for scene_id, roots in selected_scenes:
        images = collect_images(roots, args.prefer_pattern)
        selected = sample_evenly(images, args.frames_per_scene)
        scene_out_dir = args.output_root / scene_id
        scene_out_dir.mkdir(parents=True, exist_ok=True)

        for i, image_path in enumerate(selected):
            frame_id = image_path.stem
            out_name = f"{i:04d}_{image_path.name}"
            subset_path = scene_out_dir / out_name
            if args.copy_mode in {"copy", "symlink"}:
                safe_link_or_copy(image_path, subset_path, args.copy_mode)
            rows.append(
                {
                    "dataset": args.dataset,
                    "scene_id": scene_id,
                    "frame_id": frame_id,
                    "image_path": str(image_path.resolve()),
                    "subset_path": str(subset_path.resolve()),
                }
            )

        summary["scenes"][scene_id] = {
            "num_discovered_images": len(images),
            "num_selected": len(selected),
            "roots": [str(p) for p in roots],
        }

    manifest_path = args.output_root / "subset_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "scene_id", "frame_id", "image_path", "subset_path"])
        writer.writeheader()
        writer.writerows(rows)

    with (args.output_root / "scene_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote manifest with {len(rows)} rows: {manifest_path}")


if __name__ == "__main__":
    main()
