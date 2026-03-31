#!/usr/bin/env python3
"""Summarize EoMT output metadata into CSV and markdown report."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize EoMT results.")
    parser.add_argument("--outputs-root", type=Path, required=True)
    parser.add_argument("--report-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.report_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for meta_path in args.outputs_root.rglob("metadata.json"):
        with meta_path.open("r", encoding="utf-8") as f:
            m = json.load(f)
        mask_infos = m.get("mask_infos", [])
        ratios = [x.get("area_ratio", 0.0) for x in mask_infos]
        top5 = sum(sorted(ratios, reverse=True)[:5]) if ratios else 0.0

        rows.append(
            {
                "dataset": m.get("dataset", ""),
                "scene_id": m.get("scene_id", ""),
                "frame_id": m.get("frame_id", ""),
                "model_id": m.get("model_id", ""),
                "task": m.get("task", ""),
                "image_height": m.get("image_height", 0),
                "image_width": m.get("image_width", 0),
                "num_saved_masks": m.get("num_saved_masks", 0),
                "largest_mask_ratio": max(ratios) if ratios else 0.0,
                "top5_mask_coverage": top5,
                "mean_mask_ratio": float(sum(ratios) / len(ratios)) if ratios else 0.0,
                "runtime_seconds": m.get("runtime_seconds", 0.0),
                "device_used": m.get("device_used", ""),
                "fallback_used": m.get("fallback_used", False),
                "notes": "|".join(m.get("notes", [])),
            }
        )

    summary_csv = args.report_root / "summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    df = pd.DataFrame(rows)
    dataset_sizes = df.groupby("dataset").size().to_dict() if not df.empty else {}
    checkpoints = sorted(df["model_id"].unique().tolist()) if not df.empty else []

    report_md = args.report_root / "report.md"
    with report_md.open("w", encoding="utf-8") as f:
        f.write("# Local EoMT Indoor Evaluation Report\n\n")
        f.write("## 1. short experiment setup\n")
        f.write("Batch size 1 inference with Hugging Face EoMT, optional CPU fallback after CUDA OOM.\n\n")
        f.write("## 2. dataset subset sizes\n")
        f.write(json.dumps(dataset_sizes, indent=2) + "\n\n")
        f.write("## 3. checkpoints tested\n")
        for ckpt in checkpoints:
            f.write(f"- {ckpt}\n")
        f.write("\n## 4. qualitative observations\n")
        f.write("- Review contact sheets for region coherence and over/under-segmentation per scene.\n\n")
        f.write("## 5. Indoor usefulness\n")
        f.write("- Prefer models with higher top-5 mask coverage and cleaner furniture boundaries.\n\n")
        f.write("## 6. Failure modes\n")
        f.write("- Watch for oversized wall/floor masks swallowing objects and tiny fragmented masks.\n\n")
        f.write("## 7. Recommendation for next step\n")
        f.write("- Select checkpoint with best visual mask quality + coverage, then run SigLIP fusion experiments.\n")

    print(f"Wrote {summary_csv}")
    print(f"Wrote {report_md}")


if __name__ == "__main__":
    main()
