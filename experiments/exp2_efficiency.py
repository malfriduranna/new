import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn

from .datasets import build_loaders, load_manifest, ManifestEntry
from .engine import evaluate, load_checkpoint, run_training
from .models.hd_gcn_adapter import build_hd_gcn
from .models.videomae_adapter import build_videomae
from .plots import plot_learning_curve
from .utils import load_experiment_config, save_metrics_json, set_seed


def _load_entries(manifest_path: Path, video_root: Path, skeleton_root: Path) -> List[ManifestEntry]:
    return load_manifest(manifest_path, video_root, skeleton_root)


def train_models_for_subset(
    subset_entries: Sequence[ManifestEntry],
    val_entries: Sequence[ManifestEntry],
    test_entries: Sequence[ManifestEntry],
    config,
    device: torch.device,
    suffix: str,
) -> Tuple[float, Optional[float]]:
    loaders = build_loaders(
        subset_entries,
        val_entries,
        test_entries,
        sequence_length=config.hd_gcn.sequence_length,
        num_frames=config.videomae.num_frames,
        resize_to=config.videomae.image_size,
        batch_size=config.schedule.batch_size,
        num_workers=config.schedule.num_workers,
    )
    train_skel, val_skel, test_skel, train_rgb, val_rgb, test_rgb = loaders

    # HD-GCN
    hd_model = build_hd_gcn(
        num_joints=config.hd_gcn.num_joints,
        hidden_dim=config.hd_gcn.hidden_dim,
        num_classes=config.hd_gcn.num_classes,
        dropout=config.hd_gcn.dropout,
        checkpoint=config.hd_gcn.checkpoint.init_weights,
    )
    hd_best = run_training(
        model=hd_model,
        train_loader=train_skel,
        val_loader=val_skel,
        epochs=config.schedule.epochs,
        lr=config.schedule.lr,
        weight_decay=config.schedule.weight_decay,
        device=device,
        checkpoint_dir=config.output.checkpoints_dir,
        best_name=f"hd_gcn_subset_{suffix}.pt",
    )
    hd_model = build_hd_gcn(
        num_joints=config.hd_gcn.num_joints,
        hidden_dim=config.hd_gcn.hidden_dim,
        num_classes=config.hd_gcn.num_classes,
        dropout=config.hd_gcn.dropout,
    )
    hd_model = load_checkpoint(hd_model, hd_best)
    hd_loss = nn.CrossEntropyLoss()
    _, hd_metrics = evaluate(hd_model.to(device), test_skel, hd_loss, device)

    # VideoMAE
    if not config.videomae.enabled:
        return hd_metrics.accuracy, None
    vid_model = build_videomae(
        model_name_or_path=config.videomae.model_name_or_path,
        num_classes=config.videomae.num_classes,
        checkpoint=config.videomae.checkpoint.init_weights,
        eval_only=config.videomae.eval_only,
    )
    vid_best = None
    if not config.videomae.eval_only:
        vid_best = run_training(
            model=vid_model,
            train_loader=train_rgb,
            val_loader=val_rgb,
            epochs=max(1, config.schedule.epochs // 2),
            lr=config.videomae.lr,
            weight_decay=config.schedule.weight_decay,
            device=device,
            checkpoint_dir=config.output.checkpoints_dir,
            best_name=f"videomae_subset_{suffix}.pt",
        )
        vid_model = build_videomae(
            model_name_or_path=config.videomae.model_name_or_path,
            num_classes=config.videomae.num_classes,
            checkpoint=vid_best,
        )
    vid_loss = nn.CrossEntropyLoss()
    _, vid_metrics = evaluate(vid_model.to(device), test_rgb, vid_loss, device)

    return hd_metrics.accuracy, vid_metrics.accuracy


def main(config_path: Path) -> None:
    exp_set = load_experiment_config(config_path)
    config = exp_set.base
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_entries = _load_entries(config.dataset.val_manifest, config.dataset.video_root, config.dataset.skeleton_root)
    test_entries = _load_entries(config.dataset.test_manifest, config.dataset.video_root, config.dataset.skeleton_root)
    full_train_entries = _load_entries(config.dataset.train_manifest, config.dataset.video_root, config.dataset.skeleton_root)

    hd_points: List[Tuple[int, float]] = []
    vid_points: List[Tuple[int, float]] = []
    metrics_payload: Dict[str, Dict[str, float]] = {}

    for subset in exp_set.subsets:
        subset_entries = _load_entries(subset.manifest_path, config.dataset.video_root, config.dataset.skeleton_root)
        hd_acc, vid_acc = train_models_for_subset(subset_entries, val_entries, test_entries, config, device, suffix=str(subset.percentage))
        hd_points.append((subset.percentage, hd_acc))
        entry = {"hd_gcn_accuracy": hd_acc}
        if vid_acc is not None:
            vid_points.append((subset.percentage, vid_acc))
            entry["videomae_accuracy"] = vid_acc
        metrics_payload[str(subset.percentage)] = entry

    # Full dataset training
    hd_acc, vid_acc = train_models_for_subset(full_train_entries, val_entries, test_entries, config, device, suffix="full")
    hd_points.append((100, hd_acc))
    entry = {"hd_gcn_accuracy": hd_acc}
    if vid_acc is not None:
        vid_points.append((100, vid_acc))
        entry["videomae_accuracy"] = vid_acc
    metrics_payload["100"] = entry

    metrics_path = config.output.logs_dir / "exp2_data_efficiency.json"
    save_metrics_json(metrics_payload, metrics_path)

    plot_learning_curve(hd_points, title="HD-GCN Data Efficiency", output_path=config.output.plots_dir / "exp2_hdgcn.png")
    if vid_points:
        plot_learning_curve(vid_points, title="VideoMAE Data Efficiency", output_path=config.output.plots_dir / "exp2_videomae.png")

    print(f"Exp 2 complete. Metrics saved to {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp 2: Data efficiency stress test")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config")
    args = parser.parse_args()
    main(args.config)
