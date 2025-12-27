from pathlib import Path

import torch
from torch import nn
from transformers import VideoMAEForVideoClassification


class VideoMAEClassifier(nn.Module):
    def __init__(self, model_name_or_path: str, num_classes: int, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)) -> None:
        super().__init__()
        self.model = VideoMAEForVideoClassification.from_pretrained(
            model_name_or_path,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
            problem_type="single_label_classification",
        )
        self.register_buffer("mean", torch.tensor(mean).view(1, 1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 1, 3, 1, 1))

    def forward(self, videos: torch.Tensor) -> torch.Tensor:
        # videos: (batch, time, channels, height, width) in [0,1]
        pixel_values = (videos - self.mean) / self.std
        outputs = self.model(pixel_values=pixel_values)
        return outputs.logits


def build_videomae(model_name_or_path: str, num_classes: int, checkpoint: Path | None = None) -> VideoMAEClassifier:
    model = VideoMAEClassifier(model_name_or_path=model_name_or_path, num_classes=num_classes)
    if checkpoint is not None and checkpoint.exists():
        state = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state)
    return model
