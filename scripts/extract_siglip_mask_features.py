#!/usr/bin/env python3
"""Extract mask-level SigLIP features from EoMT outputs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract SigLIP features for EoMT masks.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--eomt-output-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--siglip-model-id", type=str, default="google/siglip-base-patch16-224")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "float16", "float32"], default="auto")
    parser.add_argument("--max-images", type=int, default=None)
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
    return torch.float16 if device.type == "cuda" else torch.float32


def masked_crop_feature(image: Image.Image, mask: np.ndarray, processor, model, device: torch.device) -> np.ndarray:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return np.zeros((model.config.projection_dim,), dtype=np.float32)
    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
    crop_np = np.array(image)[y1 : y2 + 1, x1 : x2 + 1].copy()
    crop_mask = mask[y1 : y2 + 1, x1 : x2 + 1]
    crop_np[crop_mask == 0] = 0
    crop = Image.fromarray(crop_np)

    inputs = processor(images=crop, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.inference_mode():
        out = model.get_image_features(**inputs)
    vec = out[0].detach().float().cpu().numpy()
    return vec


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)

    processor = AutoProcessor.from_pretrained(args.siglip_model_id)
    model = AutoModel.from_pretrained(args.siglip_model_id, torch_dtype=dtype if device.type == "cuda" else torch.float32)
    model = model.to(device)
    model.eval()

    rows = load_manifest(args.manifest)
    if args.max_images is not None:
        rows = rows[: args.max_images]

    model_dirs = [p for p in args.eomt_output_root.iterdir() if p.is_dir()]

    for row in rows:
        image_path = Path(row["image_path"])
        scene_id = row["scene_id"]
        frame_stem = image_path.stem

        for model_dir in model_dirs:
            frame_out = model_dir / scene_id / frame_stem
            masks_file = frame_out / "masks.npz"
            meta_out = frame_out / "siglip_mask_features.json"
            npy_out = frame_out / "siglip_mask_features.npy"
            if not masks_file.exists():
                continue
            if npy_out.exists() and not args.overwrite:
                continue

            image = Image.open(image_path).convert("RGB")
            masks = np.load(masks_file)["masks"]
            all_vecs = []
            entries = []

            full_inputs = processor(images=image, return_tensors="pt")
            full_inputs = {k: v.to(device) for k, v in full_inputs.items()}
            with torch.inference_mode():
                global_feature = model.get_image_features(**full_inputs)[0].detach().float().cpu().numpy()

            h, w = np.array(image).shape[:2]
            total = h * w

            for idx, mask in enumerate(masks):
                vec = masked_crop_feature(image, mask, processor, model, device)
                ys, xs = np.where(mask > 0)
                if len(xs) == 0:
                    bbox = [0, 0, 0, 0]
                    area_ratio = 0.0
                else:
                    bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
                    area_ratio = float(mask.sum() / total)
                all_vecs.append(vec)
                entries.append({"mask_index": idx, "area_ratio": area_ratio, "bbox_xyxy": bbox})

            all_vecs = np.stack(all_vecs, axis=0) if all_vecs else np.zeros((0, global_feature.shape[0]), dtype=np.float32)
            np.save(npy_out, all_vecs)
            with meta_out.open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "siglip_model_id": args.siglip_model_id,
                        "region_feature_file": str(npy_out.name),
                        "global_feature": global_feature.tolist(),
                        "num_masks": int(all_vecs.shape[0]),
                        "entries": entries,
                    },
                    f,
                    indent=2,
                )


if __name__ == "__main__":
    main()
