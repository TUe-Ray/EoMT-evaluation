#!/usr/bin/env python3
"""Helpers for running official ScanNet / ScanNet++ download scripts.

This script intentionally wraps (but does not replace) the official dataset downloaders.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from urllib.request import urlretrieve

SCANNET_SCRIPT_URL = "http://kaldir.vc.cit.tum.de/scannet/download-scannet.py"


def _run(cmd: list[str], dry_run: bool = False) -> int:
    print("$", " ".join(cmd))
    if dry_run:
        return 0
    return subprocess.run(cmd, check=False).returncode


def _require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {description}: {path}")


def _scannetpp(args: argparse.Namespace) -> int:
    script_dir = Path(args.script_dir).expanduser().resolve()
    script_path = script_dir / "download_scannetpp.py"
    template_yml = Path(args.template_yml).expanduser().resolve()

    _require_file(script_path, "ScanNet++ download script (download_scannetpp.py)")
    _require_file(template_yml, "ScanNet++ YAML config")

    token = args.token or os.getenv("SCANNETPP_TOKEN")
    if not token:
        raise ValueError(
            "No token provided. Use --token or set SCANNETPP_TOKEN in your shell."
        )

    out_dir = Path(args.download_root).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    config_out = Path(args.config_out).expanduser().resolve()
    config_out.parent.mkdir(parents=True, exist_ok=True)

    text = template_yml.read_text(encoding="utf-8")
    text = text.replace("YOUR_TOKEN_HERE", token)
    text = text.replace("DOWNLOAD_LOCATION_HERE", str(out_dir))
    config_out.write_text(text, encoding="utf-8")

    print(f"Wrote ScanNet++ config: {config_out}")
    print("Token source:", "--token" if args.token else "SCANNETPP_TOKEN env var")

    if args.run:
        rc = _run([sys.executable, str(script_path), str(config_out)], dry_run=args.dry_run)
        if rc != 0:
            print(
                "\nIf you are on Windows and see back-slashes in generated URLs, "
                "patch download_scannetpp.py to replace '\\\\' with '/'.",
                file=sys.stderr,
            )
        return rc

    print("\nNext command:")
    print(f"python {script_path} {config_out}")
    return 0


def _ensure_scannet_script(path: Path, fetch: bool, dry_run: bool) -> Path:
    if path.exists():
        return path
    if not fetch:
        raise FileNotFoundError(
            f"ScanNet script not found at {path}. Provide --scannet-script-path or use --fetch-script."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading official ScanNet script from {SCANNET_SCRIPT_URL}")
    if not dry_run:
        urlretrieve(SCANNET_SCRIPT_URL, path)
    return path


def _scannet(args: argparse.Namespace) -> int:
    script_path = Path(args.scannet_script_path).expanduser().resolve()
    script_path = _ensure_scannet_script(script_path, args.fetch_script, args.dry_run)

    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    base_cmd = [sys.executable, str(script_path), "-o", str(output_root)]

    if args.scene_id:
        rc = 0
        for sid in args.scene_id:
            cmd = base_cmd + ["--id", sid]
            if args.file_type:
                cmd += ["--type", args.file_type]
            this_rc = _run(cmd, dry_run=args.dry_run)
            rc = rc or this_rc
        return rc

    cmd = base_cmd.copy()
    if args.file_type:
        cmd += ["--type", args.file_type]
    if args.v1:
        cmd += ["--v1"]
    if args.task_data:
        cmd += ["--task_data"]
    if args.label_map:
        cmd += ["--label_map"]

    return _run(cmd, dry_run=args.dry_run)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Simple wrappers for official ScanNet/ScanNet++ download scripts"
    )
    sub = parser.add_subparsers(dest="dataset", required=True)

    spp = sub.add_parser("scannetpp", help="Prepare/run ScanNet++ official downloader")
    spp.add_argument("--script-dir", default="third_party/scannetpp_download")
    spp.add_argument(
        "--template-yml",
        default="third_party/scannetpp_download/download_scannetpp.yml",
        help="Path to official download_scannetpp.yml",
    )
    spp.add_argument(
        "--config-out",
        default="third_party/scannetpp_download/download_scannetpp.local.yml",
        help="Patched config output path",
    )
    spp.add_argument("--download-root", default="data/raw/scannetpp")
    spp.add_argument("--token", default=None, help="ScanNet++ personal token")
    spp.add_argument("--run", action="store_true", help="Run download immediately")
    spp.add_argument("--dry-run", action="store_true")
    spp.set_defaults(func=_scannetpp)

    sn = sub.add_parser("scannet", help="Run ScanNet official downloader")
    sn.add_argument(
        "--scannet-script-path",
        default="third_party/scannet/download-scannet.py",
    )
    sn.add_argument(
        "--fetch-script",
        action="store_true",
        help="Fetch official script if missing",
    )
    sn.add_argument("--output-root", default="data/raw/scannet")
    sn.add_argument(
        "--scene-id",
        action="append",
        help="Specific scene id (repeat flag for multiple scenes)",
    )
    sn.add_argument(
        "--file-type",
        default=".sens",
        help="Type filter passed to download-scannet.py --type",
    )
    sn.add_argument("--v1", action="store_true")
    sn.add_argument("--task-data", action="store_true")
    sn.add_argument("--label-map", action="store_true")
    sn.add_argument("--dry-run", action="store_true")
    sn.set_defaults(func=_scannet)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
