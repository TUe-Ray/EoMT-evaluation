#!/usr/bin/env python3
"""Run Hugging Face EoMT inference and export masks/visuals/metadata."""
from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, EomtForUniversalSegmentation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EoMT inference on manifest images.")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-id", type=str, required=True)
    parser.add_argument("--task", choices=["semantic", "panoptic"], required=True)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "float16", "float32"], default="auto")
    parser.add_argument("--semantic-shortest-edge", type=int, default=448)
    parser.add_argument("--panoptic-square-size", type=int, default=512)
    parser.add_argument("--save-topk-masks", type=int, default=8)
    parser.add_argument("--min-mask-area-ratio", type=float, default=0.01)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_dtype(dtype_arg: str, device: torch.device) -> torch.dtype:
    if dtype_arg == "float16":
        return torch.float16
    if dtype_arg == "float32":
        return torch.float32
    if device.type == "cuda":
        return torch.float16
    return torch.float32


def make_overlay(image_np: np.ndarray, seg_map: np.ndarray) -> np.ndarray:
    colors = np.random.default_rng(0).integers(0, 255, size=(max(int(seg_map.max()) + 1, 2), 3), dtype=np.uint8)
    seg_rgb = colors[seg_map % len(colors)]
    overlay = (0.6 * image_np + 0.4 * seg_rgb).astype(np.uint8)
    return overlay


def extract_masks(seg_map: np.ndarray, topk: int, min_ratio: float) -> tuple[np.ndarray, list[dict[str, float]]]:
    h, w = seg_map.shape
    total = h * w
    masks = []
    infos = []
    for seg_id in np.unique(seg_map):
        mask = seg_map == seg_id
        area = int(mask.sum())
        ratio = area / total
        if ratio < min_ratio:
            continue
        ys, xs = np.where(mask)
        bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
        masks.append(mask.astype(np.uint8))
        infos.append({"segment_id": int(seg_id), "area": area, "area_ratio": ratio, "bbox_xyxy": bbox})
    infos_masks = sorted(zip(infos, masks), key=lambda x: x[0]["area"], reverse=True)[:topk]
    if not infos_masks:
        return np.zeros((0, h, w), dtype=np.uint8), []
    infos_out = [i for i, _ in infos_masks]
    masks_out = np.stack([m for _, m in infos_masks], axis=0)
    return masks_out, infos_out


def save_contact_sheet(out_path: Path, image: np.ndarray, overlay: np.ndarray, masks: np.ndarray) -> None:
    num_masks = min(8, masks.shape[0])
    cols = 5
    rows = 2
    fig, axes = plt.subplots(rows, cols, figsize=(15, 6))
    axes = axes.flatten()
    axes[0].imshow(image)
    axes[0].set_title("original")
    axes[1].imshow(overlay)
    axes[1].set_title("overlay")
    for i in range(2, len(axes)):
        mi = i - 2
        if mi < num_masks:
            axes[i].imshow(masks[mi], cmap="gray")
            axes[i].set_title(f"mask_{mi}")
        axes[i].axis("off")
    for i in [0, 1]:
        axes[i].axis("off")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def run_one_image(
    image: Image.Image,
    processor,
    model,
    task: str,
    target_h: int,
    target_w: int,
    size_value: int,
    device: torch.device,
    dtype: torch.dtype,
):
    kwargs = {"return_tensors": "pt"}
    if task == "semantic":
        kwargs["size"] = {"shortest_edge": size_value}
    else:
        kwargs["size"] = {"shortest_edge": size_value, "longest_edge": size_value}
    inputs = processor(images=image, **kwargs)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = model(**inputs)

    if task == "semantic":
        pred = processor.post_process_semantic_segmentation(outputs, target_sizes=[(target_h, target_w)])[0]
        seg_map = pred.detach().cpu().numpy().astype(np.int32)
        extra = {"segments_info": None}
    else:
        pred = processor.post_process_panoptic_segmentation(outputs, target_sizes=[(target_h, target_w)])[0]
        seg_map = pred["segmentation"].detach().cpu().numpy().astype(np.int32)
        extra = {"segments_info": pred.get("segments_info", [])}
    return seg_map, extra


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    logging.basicConfig(filename=logs_dir / "error.log", level=logging.INFO)

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)

    processor = AutoImageProcessor.from_pretrained(args.model_id)
    model = EomtForUniversalSegmentation.from_pretrained(args.model_id, torch_dtype=dtype if device.type == "cuda" else torch.float32)
    model = model.to(device)
    model.eval()

    rows = load_manifest(args.manifest)
    if args.max_images is not None:
        rows = rows[: args.max_images]

    model_alias = args.model_id.split("/")[-1]

    for row in rows:
        dataset = row["dataset"]
        scene_id = row["scene_id"]
        frame_id = row["frame_id"]
        image_path = Path(row["image_path"])
        frame_stem = image_path.stem
        out_dir = args.output_root / model_alias / scene_id / frame_stem
        out_dir.mkdir(parents=True, exist_ok=True)

        metadata_path = out_dir / "metadata.json"
        if metadata_path.exists() and not args.overwrite:
            continue

        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)
        h, w = image_np.shape[:2]

        fallback_used = False
        notes = []
        runtime_start = time.perf_counter()

        try_sizes = [args.semantic_shortest_edge if args.task == "semantic" else args.panoptic_square_size]
        try_sizes.append(max(224, int(try_sizes[0] * 0.75)))

        seg_map = None
        extra = {}
        used_device = device.type
        for idx, size_value in enumerate(try_sizes):
            try:
                seg_map, extra = run_one_image(image, processor, model, args.task, h, w, size_value, device, dtype)
                break
            except torch.cuda.OutOfMemoryError:
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                    fallback_used = True
                    notes.append(f"oom_on_cuda_size_{size_value}")
                    logging.exception("OOM at %s", image_path)
                    if idx == len(try_sizes) - 1:
                        cpu_device = torch.device("cpu")
                        cpu_model = model.to(cpu_device)
                        seg_map, extra = run_one_image(image, processor, cpu_model, args.task, h, w, size_value, cpu_device, torch.float32)
                        used_device = "cpu"
                        model = cpu_model.to(device) if device.type == "cuda" else cpu_model
                        break
                else:
                    raise

        if seg_map is None:
            logging.error("Failed inference for %s", image_path)
            continue

        masks, mask_infos = extract_masks(seg_map, args.save_topk_masks, args.min_mask_area_ratio)
        overlay = make_overlay(image_np, seg_map)

        seg_vis = (seg_map.astype(np.float32) / max(seg_map.max(), 1) * 255).astype(np.uint8)
        Image.fromarray(image_np).save(out_dir / "original.png")
        Image.fromarray(overlay).save(out_dir / "overlay.png")
        Image.fromarray(seg_vis).save(out_dir / "segmentation.png")
        np.savez_compressed(out_dir / "masks.npz", masks=masks)
        save_contact_sheet(out_dir / "contact_sheet.png", image_np, overlay, masks)

        runtime_seconds = time.perf_counter() - runtime_start
        metadata = {
            "dataset": dataset,
            "scene_id": scene_id,
            "frame_id": frame_id,
            "image_path": str(image_path.resolve()),
            "model_id": args.model_id,
            "task": args.task,
            "image_height": h,
            "image_width": w,
            "segments_info": extra.get("segments_info"),
            "unique_label_ids": [int(x) for x in np.unique(seg_map)],
            "mask_infos": mask_infos,
            "num_saved_masks": len(mask_infos),
            "min_mask_area_ratio": args.min_mask_area_ratio,
            "runtime_seconds": runtime_seconds,
            "device_used": used_device,
            "fallback_used": fallback_used,
            "notes": notes,
        }
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    main()
