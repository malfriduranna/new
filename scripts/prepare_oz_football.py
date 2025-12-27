#!/usr/bin/env python3
"""
Utility to unpack OZ_Football_COCO.npz into per-clip skeleton tensors and CSV manifests.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


def _split_indices(n: int, train_frac: float, val_frac: float, seed: int) -> Tuple[Sequence[int], Sequence[int], Sequence[int]]:
    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)
    train_end = int(train_frac * n)
    val_end = int((train_frac + val_frac) * n)
    return indices[:train_end], indices[train_end:val_end], indices[val_end:]


def _write_manifests(entries: Dict[str, List[Dict[str, object]]], manifest_dir: Path) -> None:
    manifest_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in entries.items():
        df = pd.DataFrame(rows)
        df.to_csv(manifest_dir / f"{split}.csv", index=False)


def _write_subsets(train_df: pd.DataFrame, manifest_dir: Path, percents: Sequence[int]) -> None:
    for pct in percents:
        count = max(1, int(len(train_df) * pct / 100))
        subset = train_df.sample(n=count, random_state=42)
        subset_path = manifest_dir / f"train_{pct}.csv"
        subset.to_csv(subset_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare OZ Football COCO skeleton manifests.")
    parser.add_argument("--source", type=Path, default=Path("data/skeletons/OZ_Football_COCO.npz"), help="Path to OZ_Football_COCO.npz")
    parser.add_argument("--skeleton-root", type=Path, default=Path("data/skeletons"), help="Root directory for skeleton assets")
    parser.add_argument("--skeleton-subdir", type=str, default="oz_football", help="Sub-directory under skeleton root to store per-clip tensors")
    parser.add_argument("--manifest-dir", type=Path, default=Path("data/manifests"), help="Where to write CSV manifests")
    parser.add_argument("--label-names", type=Path, default=Path("configs/label_names.txt"), help="Label names file to overwrite")
    parser.add_argument("--video-placeholder", type=str, default="dummy.mp4", help="Video path stub to record in manifests")
    parser.add_argument("--train-frac", type=float, default=0.7, help="Training fraction")
    parser.add_argument("--val-frac", type=float, default=0.15, help="Validation fraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling before splitting")
    parser.add_argument("--subset-percents", type=int, nargs="*", default=(10, 25, 50), help="Percents for subset manifests")
    args = parser.parse_args()

    if not args.source.exists():
        raise FileNotFoundError(f"Source NPZ not found: {args.source}")

    skeleton_dir = args.skeleton_root / args.skeleton_subdir
    skeleton_dir.mkdir(parents=True, exist_ok=True)
    npz = np.load(args.source, allow_pickle=True)
    sequences = np.squeeze(npz["x_train"], axis=-1).astype("float32")
    labels = np.argmax(npz["y_train"], axis=1).astype(int)

    train_idx, val_idx, test_idx = _split_indices(len(sequences), args.train_frac, args.val_frac, args.seed)
    membership = {idx: "train" for idx in train_idx}
    membership.update({idx: "val" for idx in val_idx})
    membership.update({idx: "test" for idx in test_idx})

    entries: Dict[str, List[Dict[str, object]]] = {"train": [], "val": [], "test": []}
    rel_prefix = Path(args.skeleton_subdir)

    for idx, seq in enumerate(sequences):
        split = membership.get(idx, "test")
        fname = f"oz_{idx:04d}.npy"
        out_path = skeleton_dir / fname
        np.save(out_path, seq)
        entries[split].append(
            {
                "clip_id": f"oz_{idx:04d}",
                "video_path": args.video_placeholder,
                "skeleton_path": str(rel_prefix / fname).replace("\\", "/"),
                "label": int(labels[idx]),
            }
        )

    _write_manifests(entries, args.manifest_dir)
    train_df = pd.DataFrame(entries["train"])
    _write_subsets(train_df, args.manifest_dir, args.subset_percents)

    label_map = npz["labels"].tolist()
    ordered = [name for name, _ in sorted(label_map.items(), key=lambda kv: kv[1])]
    args.label_names.write_text("\n".join(ordered) + "\n")

    print(f"Wrote skeleton clips to {skeleton_dir} and manifests to {args.manifest_dir}")


if __name__ == "__main__":
    main()
