from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision


@dataclass
class ManifestEntry:
    clip_id: str
    video_path: Path
    skeleton_path: Path
    label: int


class SkeletonDataset(Dataset[Tuple[torch.Tensor, int]]):
    def __init__(
        self,
        entries: Sequence[ManifestEntry],
        sequence_length: int,
        noise_sigma: float = 0.0,
    ) -> None:
        self.entries = list(entries)
        self.sequence_length = sequence_length
        self.noise_sigma = noise_sigma

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        entry = self.entries[idx]
        skeleton = np.load(entry.skeleton_path)
        if skeleton.ndim == 4:
            skeleton = np.squeeze(skeleton, axis=-1)
        if skeleton.ndim != 3:
            raise ValueError(f"Expected skeleton tensor with 3 dimensions (T, joints, coords); got shape {skeleton.shape}")
        if skeleton.shape[-1] > 2:
            skeleton = skeleton[..., :2]
        if skeleton.shape[0] < self.sequence_length:
            pad = np.repeat(skeleton[-1:], self.sequence_length - skeleton.shape[0], axis=0)
            skeleton = np.concatenate([skeleton, pad], axis=0)
        skeleton = skeleton[: self.sequence_length]
        tensor = torch.tensor(skeleton, dtype=torch.float32)
        if self.noise_sigma > 0:
            noise = torch.randn_like(tensor) * self.noise_sigma
            tensor = tensor + noise
        return tensor, entry.label


class RGBDataset(Dataset[Tuple[torch.Tensor, int]]):
    def __init__(
        self,
        entries: Sequence[ManifestEntry],
        num_frames: int,
        resize_to: int,
        center_crop: bool = True,
        downsample_to: Optional[int] = None,
    ) -> None:
        self.entries = list(entries)
        self.num_frames = num_frames
        self.resize_to = resize_to
        self.center_crop = center_crop
        self.downsample_to = downsample_to

        transforms_list: List[Callable[[torch.Tensor], torch.Tensor]] = [transforms.Resize(resize_to)]
        if center_crop:
            transforms_list.append(transforms.CenterCrop(resize_to))
        self.frame_transform = transforms.Compose(transforms_list)

    def __len__(self) -> int:
        return len(self.entries)

    def _temporal_sample(self, frames: torch.Tensor) -> torch.Tensor:
        total_frames = frames.shape[0]
        if total_frames >= self.num_frames:
            idxs = torch.linspace(0, total_frames - 1, steps=self.num_frames).long()
            return frames[idxs]
        repeat = self.num_frames - total_frames
        padding = frames[-1:].repeat(repeat, 1, 1, 1)
        return torch.cat([frames, padding], dim=0)

    def _degrade(self, frames: torch.Tensor) -> torch.Tensor:
        if self.downsample_to is None:
            return frames
        down = transforms.Resize(self.downsample_to)
        up = transforms.Resize(self.resize_to)
        degraded = down(frames)
        return up(degraded)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        entry = self.entries[idx]
        video, _, _ = torchvision.io.read_video(str(entry.video_path), pts_unit="sec")
        # video: T x H x W x C
        video = video.permute(0, 3, 1, 2)  # T, C, H, W
        video = self._temporal_sample(video)
        frames = torch.stack([self.frame_transform(frame) for frame in video])
        frames = self._degrade(frames)
        frames = frames.float() / 255.0
        return frames, entry.label


def _resolve_path(root: Path, maybe_relative: str) -> Path:
    path = Path(maybe_relative)
    if path.is_absolute():
        return path
    return root / path


def load_manifest(manifest_path: Path, video_root: Path, skeleton_root: Path) -> List[ManifestEntry]:
    df = pd.read_csv(manifest_path)
    required_cols = {"clip_id", "video_path", "skeleton_path", "label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Manifest {manifest_path} is missing columns: {missing}")

    entries: List[ManifestEntry] = []
    for _, row in df.iterrows():
        entries.append(
            ManifestEntry(
                clip_id=str(row["clip_id"]),
                video_path=_resolve_path(video_root, row["video_path"]),
                skeleton_path=_resolve_path(skeleton_root, row["skeleton_path"]),
                label=int(row["label"]),
            )
        )
    return entries


def build_loaders(
    train_entries: Sequence[ManifestEntry],
    val_entries: Sequence[ManifestEntry],
    test_entries: Sequence[ManifestEntry],
    *,
    sequence_length: int,
    num_frames: int,
    resize_to: int,
    batch_size: int,
    num_workers: int,
    noise_sigma: float = 0.0,
    downsample_to: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader, DataLoader, DataLoader]:
    train_skel = SkeletonDataset(train_entries, sequence_length=sequence_length, noise_sigma=noise_sigma)
    val_skel = SkeletonDataset(val_entries, sequence_length=sequence_length, noise_sigma=0.0)
    test_skel = SkeletonDataset(test_entries, sequence_length=sequence_length, noise_sigma=noise_sigma)

    train_rgb = RGBDataset(train_entries, num_frames=num_frames, resize_to=resize_to, center_crop=True, downsample_to=downsample_to)
    val_rgb = RGBDataset(val_entries, num_frames=num_frames, resize_to=resize_to, center_crop=True, downsample_to=downsample_to)
    test_rgb = RGBDataset(test_entries, num_frames=num_frames, resize_to=resize_to, center_crop=True, downsample_to=downsample_to)

    def loader(dataset: Dataset[Tuple[torch.Tensor, int]], shuffle: bool) -> DataLoader:
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)

    return (
        loader(train_skel, shuffle=True),
        loader(val_skel, shuffle=False),
        loader(test_skel, shuffle=False),
        loader(train_rgb, shuffle=True),
        loader(val_rgb, shuffle=False),
        loader(test_rgb, shuffle=False),
    )
