from pathlib import Path
import torch
from torch import nn


class HDGCNClassifier(nn.Module):
    """Lightweight stand-in for HD-GCN classifier.

    Replace this with a full HD-GCN implementation if available. The model
    consumes skeleton sequences shaped as (batch, time, joints, coords).
    """

    def __init__(self, num_joints: int, hidden_dim: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.num_joints = num_joints
        self.backbone = nn.GRU(input_size=num_joints * 2, hidden_size=hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, time, joints, coords)
        b, t, j, c = x.shape
        if j != self.num_joints:
            raise ValueError(f"Expected {self.num_joints} joints, received {j}")
        x_flat = x.view(b, t, j * c)
        outputs, _ = self.backbone(x_flat)
        pooled = outputs.mean(dim=1)
        logits = self.fc(self.dropout(pooled))
        return logits


def build_hd_gcn(num_joints: int, hidden_dim: int, num_classes: int, dropout: float, checkpoint: Path | None = None) -> HDGCNClassifier:
    model = HDGCNClassifier(num_joints=num_joints, hidden_dim=hidden_dim, num_classes=num_classes, dropout=dropout)
    if checkpoint is not None and checkpoint.exists():
        state = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state)
    return model
