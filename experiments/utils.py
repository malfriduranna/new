import json
import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml

import warnings

warnings.filterwarnings(
    "ignore",
    message="The video decoding and encoding capabilities of torchvision are deprecated",
    category=UserWarning,
)

from .config import (
    DatasetPaths,
    ExperimentConfig,
    FusionSearchConfig,
    HDGCNConfig,
    OutputConfig,
    RobustnessNoiseConfig,
    RobustnessResolutionConfig,
    SemanticGroups,
    SubsetSetting,
    TrainingSchedule,
    VideoMAEConfig,
    ExperimentSetConfig,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _dataset_paths(data: Dict[str, Any]) -> DatasetPaths:
    return DatasetPaths(
        train_manifest=Path(data["train_manifest"]),
        val_manifest=Path(data["val_manifest"]),
        test_manifest=Path(data["test_manifest"]),
        video_root=Path(data["video_root"]),
        skeleton_root=Path(data["skeleton_root"]),
    )


def _hdgcn_config(data: Dict[str, Any]) -> HDGCNConfig:
    return HDGCNConfig(
        num_joints=int(data["num_joints"]),
        num_classes=int(data["num_classes"]),
        sequence_length=int(data["sequence_length"]),
        hidden_dim=int(data.get("hidden_dim", 256)),
        dropout=float(data.get("dropout", 0.2)),
    )


def _videomae_config(data: Dict[str, Any]) -> VideoMAEConfig:
    return VideoMAEConfig(
        num_frames=int(data["num_frames"]),
        image_size=int(data["image_size"]),
        num_classes=int(data["num_classes"]),
        patch_size=int(data.get("patch_size", 16)),
        enabled=bool(data.get("enabled", True)),
    )


def _schedule(data: Dict[str, Any]) -> TrainingSchedule:
    return TrainingSchedule(
        epochs=int(data["epochs"]),
        batch_size=int(data["batch_size"]),
        lr=float(data["lr"]),
        weight_decay=float(data["weight_decay"]),
        warmup_epochs=int(data.get("warmup_epochs", 5)),
        num_workers=int(data.get("num_workers", 8)),
        gradient_accumulation_steps=int(data.get("gradient_accumulation_steps", 1)),
    )


def load_experiment_config(config_path: Path) -> ExperimentSetConfig:
    data = yaml.safe_load(config_path.read_text())
    base = data["base"]
    output_root_env = os.environ.get("ACTION_EXP_OUTPUT_ROOT")
    output_root = Path(output_root_env) if output_root_env else Path(base["output_root"])
    exp_config = ExperimentConfig(
        dataset=_dataset_paths(base["dataset"]),
        hd_gcn=_hdgcn_config(base["hd_gcn"]),
        videomae=_videomae_config(base["videomae"]),
        schedule=_schedule(base["schedule"]),
        output=OutputConfig.from_root(output_root),
        seed=int(base.get("seed", 42)),
        label_names_path=Path(base["label_names_path"]) if "label_names_path" in base else None,
    )

    subsets = [SubsetSetting(percentage=int(s["percentage"]), manifest_path=Path(s["manifest_path"])) for s in data["subsets"]]
    semantic_groups = SemanticGroups(
        ball_interaction=list(data["semantic_groups"]["ball_interaction"]),
        body_motion=list(data["semantic_groups"]["body_motion"]),
    )
    fusion = FusionSearchConfig(alphas=[float(a) for a in data["fusion"]["alphas"]])
    robustness_noise = RobustnessNoiseConfig(sigmas=[float(s) for s in data["robustness_noise"]["sigmas"]])
    robustness_resolution = RobustnessResolutionConfig(resolutions=[int(r) for r in data["robustness_resolution"]["resolutions"]])

    return ExperimentSetConfig(
        base=exp_config,
        subsets=subsets,
        semantic_groups=semantic_groups,
        fusion=fusion,
        robustness_noise=robustness_noise,
        robustness_resolution=robustness_resolution,
    )


def save_metrics_json(metrics: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(metrics, f, indent=2)
