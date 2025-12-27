from pathlib import Path
from typing import List, Sequence, Tuple
import warnings

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision
from transformers import VideoMAEForVideoClassification


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _default_extensions() -> Tuple[str, ...]:
    return (".mp4", ".mkv", ".avi", ".mov")


class SoccerNetRGBDataset(Dataset[Tuple[torch.Tensor, int]]):
    def __init__(
        self,
        root: Path,
        split: str,
        class_names: Sequence[str],
        *,
        num_frames: int = 16,
        frame_stride: int = 4,
        image_size: int = 224,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.class_names = list(class_names)
        self.num_frames = num_frames
        self.frame_stride = frame_stride
        self.image_size = image_size
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        self.samples: List[Tuple[Path, int]] = []

        for cls in self.class_names:
            class_dir = self.root / split / cls
            if not class_dir.exists():
                continue
            for ext in _default_extensions():
                for path in sorted(class_dir.glob(f"*{ext}")):
                    self.samples.append((path, self.class_to_idx[cls]))

        if not self.samples:
            raise ValueError(f"No video clips found under {self.root}/{split}. Did you run the curation script?")

        self.frame_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size), antialias=True),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def _sample_indices(self, num_total_frames: int) -> torch.Tensor:
        wanted = torch.arange(0, self.num_frames * self.frame_stride, step=self.frame_stride)
        if num_total_frames <= 0:
            return torch.zeros(self.num_frames, dtype=torch.long)
        clipped = torch.clamp(wanted, max=max(num_total_frames - 1, 0))
        return clipped

    def _decode(self, path: Path) -> torch.Tensor:
        try:
            video, _, _ = torchvision.io.read_video(str(path), pts_unit="sec")
        except Exception as exc:  # pragma: no cover - defensive fallback
            warnings.warn(f"Failed to decode {path}: {exc}. Using blank clip.")
            video = torch.zeros(
                (self.num_frames * self.frame_stride, self.image_size, self.image_size, 3),
                dtype=torch.uint8,
            )
        if video.ndim != 4 or video.shape[-1] != 3:
            warnings.warn(f"Unexpected video shape {video.shape} for {path}; replacing with blank clip.")
            video = torch.zeros(
                (self.num_frames * self.frame_stride, self.image_size, self.image_size, 3),
                dtype=torch.uint8,
            )
        return video  # T, H, W, C

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        video = self._decode(path)
        frame_indices = self._sample_indices(video.shape[0])
        selected = video[frame_indices]  # num_frames, H, W, C

        frames: List[torch.Tensor] = []
        for frame in selected:
            frame_chw = frame.permute(2, 0, 1).float() / 255.0
            frames.append(self.frame_transform(frame_chw))
        clip = torch.stack(frames, dim=1)  # C, T, H, W
        return clip, label

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    def label_to_id(self, label_name: str) -> int:
        return self.class_to_idx[label_name]

    def id_to_label(self, idx: int) -> str:
        return self.class_names[idx]


def build_videomae_model(class_names: Sequence[str], device: torch.device) -> VideoMAEForVideoClassification:
    id2label = {idx: name for idx, name in enumerate(class_names)}
    label2id = {name: idx for idx, name in id2label.items()}
    model = VideoMAEForVideoClassification.from_pretrained(
        "MCG-NJU/videomae-base-finetuned-kinetics",
        num_labels=len(class_names),
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
    )
    return model.to(device)


def build_dataloaders(
    dataset_root: Path,
    class_names: Sequence[str],
    *,
    batch_size: int,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    train_ds = SoccerNetRGBDataset(dataset_root, "train", class_names)
    val_ds = SoccerNetRGBDataset(dataset_root, "valid", class_names)

    def make_loader(ds: Dataset[Tuple[torch.Tensor, int]], shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )

    return make_loader(train_ds, True), make_loader(val_ds, False)
