"""Write train.json and val.json under ScanNet-processed/.

Format mirrors DL3DV's ``[[subset, hash], ...]`` convention so existing tooling
(parse_results.py, per-VFM feature paths) keeps working unchanged. For ScanNet
we set ``subset = split`` (i.e. ``train`` or ``val``) and ``hash = scene_id``.

Example:
    python -m probing_vlm_vgm.data.processing.scannet.create_split \\
        --processed_root data/ScanNet/ScanNet-processed
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


def scenes_with_metadata(processed_split_dir: Path) -> List[str]:
    """Return sorted scene IDs that have a complete metadata.sft."""
    if not processed_split_dir.is_dir():
        return []
    out = []
    for d in sorted(processed_split_dir.iterdir()):
        if not d.is_dir():
            continue
        if (d / "metadata.sft").is_file():
            out.append(d.name)
    return out


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--processed_root",
        type=Path,
        default=Path("data/ScanNet/ScanNet-processed"),
    )
    p.add_argument("--splits", nargs="+", default=["train", "val"])
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    for split in args.splits:
        split_dir = args.processed_root / split
        scenes = scenes_with_metadata(split_dir)
        # Pair format [[subset, hash], ...] — subset == split here for ScanNet.
        pairs = [[split, s] for s in scenes]
        out_path = args.processed_root / f"{split}.json"
        with out_path.open("w") as f:
            json.dump(pairs, f, indent=2)
        logger.info(f"Wrote {out_path} with {len(pairs)} scenes.")


if __name__ == "__main__":
    main()
