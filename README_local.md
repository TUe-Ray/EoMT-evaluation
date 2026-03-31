# Local, Laptop-Friendly EoMT Evaluation Pipeline

This repository is organized for **inference-only** EoMT experiments on a single GPU (e.g., RTX 3050), with robust CPU fallback and batch size 1.

## 1) Environment setup

```bash
conda create -n eomt_local python=3.11 -y
conda activate eomt_local
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install \
  transformers==4.57.3 \
  accelerate \
  safetensors \
  huggingface_hub \
  pillow \
  matplotlib \
  opencv-python \
  numpy \
  pandas \
  tqdm \
  scikit-image \
  pycocotools
```

Clone official EoMT repo for reference only (no training setup required):

```bash
git clone https://github.com/tue-mps/eomt.git third_party/eomt_official
```

## 2) Download ScanNet / ScanNet++ with simple commands

> Use only the **official** download scripts. This repo provides a helper wrapper: `scripts/download_data_helper.py` to keep setup simple and reproducible.

### 2.1 ScanNet++ (small/targeted or full)

1. Download and unzip the official ScanNet++ download package into `third_party/scannetpp_download/` so it contains:
   - `download_scannetpp.py`
   - `download_scannetpp.yml`

2. Set your token in shell (recommended, avoids writing it in shell history repeatedly):

```bash
export SCANNETPP_TOKEN='YOUR_PERSONAL_TOKEN'
```

3. Create a local config with token + download path patched in:

```bash
python scripts/download_data_helper.py scannetpp \
  --script-dir third_party/scannetpp_download \
  --template-yml third_party/scannetpp_download/download_scannetpp.yml \
  --config-out third_party/scannetpp_download/download_scannetpp.local.yml \
  --download-root data/raw/scannetpp
```

4. Edit `third_party/scannetpp_download/download_scannetpp.local.yml` to download only a few scenes/assets first (recommended), then run:

```bash
python scripts/download_data_helper.py scannetpp \
  --script-dir third_party/scannetpp_download \
  --template-yml third_party/scannetpp_download/download_scannetpp.yml \
  --config-out third_party/scannetpp_download/download_scannetpp.local.yml \
  --download-root data/raw/scannetpp \
  --run
```

### 2.2 ScanNet (download a few scenes quickly)

Download only selected scenes (example: 2 scenes, `.sens` only):

```bash
python scripts/download_data_helper.py scannet \
  --fetch-script \
  --output-root data/raw/scannet \
  --scene-id scene0000_00 \
  --scene-id scene0001_00 \
  --file-type .sens
```

If you already have `download-scannet.py`, omit `--fetch-script` and point to it via `--scannet-script-path`.

### 2.3 Notes

- Your ScanNet++ token expires on **2026-05-23 23:59:59 +0200**; download before that date.
- Do **not** commit tokens or private dataset configs to git.
- On Windows ScanNet++, if generated URLs contain backslashes, patch the official script to replace `\\` with `/` as recommended by the dataset authors.

## 3) Dataset path expectations

- ScanNet: `<root>/<scene_id>/color/*.jpg` preferred.
  - If only `.sens` is present, discovery/subset scripts warn and skip heavy conversion.
- ScanNet++: `<root>/<scene_id>/dslr/resized_undistorted_images/*`
- ARKitScenes: `<root>/<scene_or_video>/lowres_wide/*` or `vga_wide/*`

All roots are configurable via CLI.

## 4) Subset generation commands

```bash
python scripts/discover_frames.py --dataset scannet --input-root data/raw/scannet
python scripts/discover_frames.py --dataset scannetpp --input-root data/raw/scannetpp
python scripts/discover_frames.py --dataset arkitscenes --input-root data/raw/arkitscenes

python scripts/build_subsets.py --dataset scannet --input-root data/raw/scannet --output-root data/subsets/scannet --num-scenes 2 --frames-per-scene 10 --seed 42 --copy-mode manifest
python scripts/build_subsets.py --dataset scannetpp --input-root data/raw/scannetpp --output-root data/subsets/scannetpp --num-scenes 2 --frames-per-scene 10 --seed 42 --copy-mode manifest
python scripts/build_subsets.py --dataset arkitscenes --input-root data/raw/arkitscenes --output-root data/subsets/arkitscenes --num-scenes 2 --frames-per-scene 10 --seed 42 --copy-mode manifest
```

## 5) Inference commands

Smoke test (small COCO panoptic):

```bash
python scripts/run_eomt_inference.py \
  --input-root data/subsets/scannet \
  --manifest data/subsets/scannet/subset_manifest.csv \
  --output-root outputs/scannet \
  --model-id tue-mps/coco_panoptic_eomt_small_640_2x \
  --task panoptic \
  --max-images 4 \
  --device auto \
  --dtype auto \
  --panoptic-square-size 512 \
  --save-topk-masks 8 \
  --min-mask-area-ratio 0.01
```

ADE20K semantic comparison:

```bash
python scripts/run_eomt_inference.py \
  --input-root data/subsets/scannet \
  --manifest data/subsets/scannet/subset_manifest.csv \
  --output-root outputs/scannet \
  --model-id tue-mps/ade20k_semantic_eomt_large_512 \
  --task semantic \
  --max-images 4 \
  --device auto \
  --dtype auto \
  --semantic-shortest-edge 448 \
  --save-topk-masks 8 \
  --min-mask-area-ratio 0.01
```

## 6) SigLIP extraction commands

```bash
python scripts/extract_siglip_mask_features.py \
  --manifest data/subsets/scannet/subset_manifest.csv \
  --eomt-output-root outputs/scannet \
  --output-root outputs/scannet \
  --siglip-model-id google/siglip-base-patch16-224 \
  --device auto \
  --dtype auto \
  --max-images 20
```

## 7) Report generation commands

```bash
python scripts/summarize_results.py --outputs-root outputs --report-root outputs/reports
```

## 8) Troubleshooting

- **CUDA OOM**: inference auto-retries smaller size, then CPU fallback per-image.
- **No scenes discovered**: verify expected folder conventions and input root.
- **No `masks.npz`**: check metadata and logs/error.log for model/runtime failures.
- **Slow SigLIP extraction**: expected with crop-based fallback region encoding.

## 9) Expected output directory structure

```text
outputs/
  <dataset>/
    <model_alias>/
      <scene_id>/
        <frame_stem>/
          original.png
          overlay.png
          segmentation.png
          masks.npz
          contact_sheet.png
          metadata.json
          siglip_mask_features.npy
          siglip_mask_features.json
  reports/
    summary.csv
    report.md
```

## Suggested first commands

1. Discover one dataset root.
2. Build a tiny subset.
3. Run small COCO panoptic checkpoint on 4 images.
4. Inspect `overlay.png`, `masks.npz`, `metadata.json`, `contact_sheet.png`.
5. Run ADE20K semantic checkpoint on the same 4 images.
6. Generate report.
