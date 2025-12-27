from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class DatasetPaths:
    train_manifest: Path
    val_manifest: Path
    test_manifest: Path
    video_root: Path
    skeleton_root: Path


@dataclass
class TrainingSchedule:
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    warmup_epochs: int = 5
    num_workers: int = 8
    gradient_accumulation_steps: int = 1


@dataclass
class ModelCheckpoint:
    init_weights: Optional[Path] = None
    best_checkpoint: Optional[Path] = None


@dataclass
class HDGCNConfig:
    num_joints: int
    num_classes: int
    sequence_length: int
    hidden_dim: int = 256
    dropout: float = 0.2
    checkpoint: ModelCheckpoint = field(default_factory=ModelCheckpoint)


@dataclass
class VideoMAEConfig:
    num_frames: int
    image_size: int
    num_classes: int
    patch_size: int = 16
    checkpoint: ModelCheckpoint = field(default_factory=ModelCheckpoint)
    lr: float = 3e-4


@dataclass
class OutputConfig:
    root_dir: Path
    plots_dir: Path
    logs_dir: Path
    checkpoints_dir: Path

    @staticmethod
    def from_root(root_dir: Path) -> "OutputConfig":
        plots = root_dir / "plots"
        logs = root_dir / "logs"
        ckpts = root_dir / "checkpoints"
        plots.mkdir(parents=True, exist_ok=True)
        logs.mkdir(parents=True, exist_ok=True)
        ckpts.mkdir(parents=True, exist_ok=True)
        return OutputConfig(root_dir=root_dir, plots_dir=plots, logs_dir=logs, checkpoints_dir=ckpts)


@dataclass
class ExperimentConfig:
    dataset: DatasetPaths
    hd_gcn: HDGCNConfig
    videomae: VideoMAEConfig
    schedule: TrainingSchedule
    output: OutputConfig
    seed: int = 42
    label_names_path: Optional[Path] = None


@dataclass
class SubsetSetting:
    percentage: int
    manifest_path: Path


@dataclass
class SemanticGroups:
    ball_interaction: List[str]
    body_motion: List[str]


@dataclass
class FusionSearchConfig:
    alphas: List[float] = field(default_factory=lambda: [i / 10.0 for i in range(11)])


@dataclass
class RobustnessNoiseConfig:
    sigmas: List[float]


@dataclass
class RobustnessResolutionConfig:
    resolutions: List[int]


@dataclass
class ExperimentSetConfig:
    base: ExperimentConfig
    subsets: List[SubsetSetting]
    semantic_groups: SemanticGroups
    fusion: FusionSearchConfig
    robustness_noise: RobustnessNoiseConfig
    robustness_resolution: RobustnessResolutionConfig
