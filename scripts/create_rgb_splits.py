import argparse
import shutil
from pathlib import Path
from typing import Iterable, Sequence

DEFAULT_CLASSES = ["Goal", "Foul", "Corner", "Shots on target", "Shots off target"]


def _iter_videos(class_dir: Path) -> Iterable[Path]:
    return sorted(class_dir.rglob("*.mp4"))


def _copy_subset(src: Path, dst_root: Path, count: int | None) -> int:
    copied = 0
    for video in _iter_videos(src):
        if count is not None and copied >= count:
            break
        rel = video.relative_to(src)
        dst = dst_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            continue
        shutil.copy2(video, dst)
        copied += 1
    return copied


def create_splits(dataset_root: Path, classes: Sequence[str], val_count: int, test_count: int) -> None:
    for cls in classes:
        train_dir = dataset_root / "train" / cls
        if not train_dir.exists():
            print(f"Skipping {cls}: no videos in {train_dir}")
            continue
        val_dir = dataset_root / "valid" / cls
        test_dir = dataset_root / "test" / cls
        val_copied = _copy_subset(train_dir, val_dir, val_count)
        test_copied = _copy_subset(train_dir, test_dir, test_count)
        print(f"{cls}: copied {val_copied} to valid, {test_copied} to test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Copy a few clips per class into valid/test splits.")
    parser.add_argument("--dataset-root", type=Path, default=Path("data/soccernet_rgb"), help="Root of curated dataset.")
    parser.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES, help="Class names to process.")
    parser.add_argument("--val-count", type=int, default=2, help="Number of clips to copy per class into valid.")
    parser.add_argument("--test-count", type=int, default=2, help="Number of clips to copy per class into test.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    create_splits(args.dataset_root, args.classes, args.val_count, args.test_count)


if __name__ == "__main__":
    main()
