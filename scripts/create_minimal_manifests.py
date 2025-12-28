"""
Create placeholder train/val/test manifest CSVs for quick evaluation.

Each manifest row contains `clip_id,video_path,skeleton_path,label`.
By default the script:
- creates a dummy RGB clip at data/videos/dummy.mp4 (if missing)
- creates a dummy skeleton array at data/skeletons/dummy.npy (if missing)
- writes one-row train/val/test CSVs under data/manifests/

You can override paths and labels via CLI flags.
"""

import argparse
import csv
from pathlib import Path
from typing import Iterable, List

import numpy as np
import torch
from torchvision.io import write_video


def _ensure_video(path: Path, *, num_frames: int = 16, size: int = 64, fps: int = 8) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    frames = torch.zeros((num_frames, size, size, 3), dtype=torch.uint8)
    write_video(str(path), frames, fps=fps)


def _ensure_skeleton(path: Path, *, num_frames: int = 64, num_joints: int = 17) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    data = np.zeros((num_frames, num_joints, 2), dtype=np.float32)
    np.save(path, data)


def _relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _write_manifest(manifest_path: Path, rows: Iterable[List[str]]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["clip_id", "video_path", "skeleton_path", "label"])
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create minimal manifests for pretrained VideoMAE evaluation.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/manifests"), help="Directory to write train/val/test CSVs.")
    parser.add_argument("--video-path", type=Path, default=Path("data/videos/dummy.mp4"), help="Path to RGB clip used in all splits.")
    parser.add_argument(
        "--skeleton-path",
        type=Path,
        default=Path("data/skeletons/dummy.npy"),
        help="Path to skeleton .npy used in all splits.",
    )
    parser.add_argument("--clip-id", type=str, default="dummy_clip", help="Clip identifier used in the manifests.")
    parser.add_argument("--label", type=int, default=0, help="Label index for the clip.")
    parser.add_argument("--video-root", type=Path, default=None, help="Root used to relativize video_path entries (defaults to video directory).")
    parser.add_argument(
        "--skeleton-root",
        type=Path,
        default=None,
        help="Root used to relativize skeleton_path entries (defaults to skeleton directory).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Which split CSVs to create. Each gets identical rows.",
    )
    parser.add_argument("--rows-per-split", type=int, default=1, help="How many duplicate rows to write per split.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _ensure_video(args.video_path)
    _ensure_skeleton(args.skeleton_path)

    video_root = args.video_root or args.video_path.parent
    skeleton_root = args.skeleton_root or args.skeleton_path.parent

    video_entry = _relative_or_absolute(args.video_path, video_root)
    skeleton_entry = _relative_or_absolute(args.skeleton_path, skeleton_root)

    row = [args.clip_id, video_entry, skeleton_entry, str(args.label)]
    rows = [row for _ in range(max(1, args.rows_per_split))]

    for split in args.splits:
        manifest_path = args.output_dir / f"{split}.csv"
        _write_manifest(manifest_path, rows)
        print(f"[ok] wrote {manifest_path} with {len(rows)} row(s)")


if __name__ == "__main__":
    main()
