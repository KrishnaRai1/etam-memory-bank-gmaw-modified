from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.benchmark.processed_dataset_loader import save_processed_dataset_inventory


def main() -> None:
    parser = argparse.ArgumentParser(description="Discover processed HSV benchmark datasets under a dataset root.")
    parser.add_argument("--dataset-root", default=None, help="Root folder containing video_id/timestamp/final directories")
    parser.add_argument("--out", default="outputs/processed_dataset_inventory.json", help="Output JSON path")
    args = parser.parse_args()

    dataset_root = args.dataset_root
    if not dataset_root:
        pipeline_cfg = REPO_ROOT / "configs" / "pipeline.yaml"
        if pipeline_cfg.exists():
            cfg = yaml.safe_load(pipeline_cfg.read_text(encoding="utf-8")) or {}
            dataset_root = cfg.get("data", {}).get("reference_root", "New_experiments_v3_final")
        else:
            dataset_root = "New_experiments_v3_final"

    report = save_processed_dataset_inventory(dataset_root, output_path=args.out)
    print(f"Videos found: {report['videos_found']}")
    for video in report.get("videos", []):
        print(f"- {video['video_id']}: latest_timestamp={video['timestamp']}")
        print(f"  reference_dir={video['reference_dir']}")
        print(f"  reference_files_found={','.join(video.get('found_files', [])) or 'none'}")
        print(f"  missing_files={','.join(video.get('missing_files', [])) or 'none'}")
    print(f"Inventory written to {args.out}")


if __name__ == "__main__":
    main()
