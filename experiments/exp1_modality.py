import argparse
from pathlib import Path

import torch
from torch import nn

from .config import ExperimentConfig
from .datasets import build_loaders, load_manifest
from .engine import evaluate, load_checkpoint, run_training
from .models.hd_gcn_adapter import build_hd_gcn
from .models.videomae_adapter import build_videomae
from .plots import plot_confusion
from .utils import load_experiment_config, save_metrics_json, set_seed


def prepare_loaders(config: ExperimentConfig):
    train_entries = load_manifest(config.dataset.train_manifest, config.dataset.video_root, config.dataset.skeleton_root)
    val_entries = load_manifest(config.dataset.val_manifest, config.dataset.video_root, config.dataset.skeleton_root)
    test_entries = load_manifest(config.dataset.test_manifest, config.dataset.video_root, config.dataset.skeleton_root)

    return build_loaders(
        train_entries,
        val_entries,
        test_entries,
        sequence_length=config.hd_gcn.sequence_length,
        num_frames=config.videomae.num_frames,
        resize_to=config.videomae.image_size,
        batch_size=config.schedule.batch_size,
        num_workers=config.schedule.num_workers,
    )


def run_hd_gcn(config: ExperimentConfig, device: torch.device, loaders):
    train_skel, val_skel, test_skel, _, _, _ = loaders
    model = build_hd_gcn(
        num_joints=config.hd_gcn.num_joints,
        hidden_dim=config.hd_gcn.hidden_dim,
        num_classes=config.hd_gcn.num_classes,
        dropout=config.hd_gcn.dropout,
        checkpoint=config.hd_gcn.checkpoint.init_weights,
    )
    best_path = run_training(
        model=model,
        train_loader=train_skel,
        val_loader=val_skel,
        epochs=config.schedule.epochs,
        lr=config.schedule.lr,
        weight_decay=config.schedule.weight_decay,
        device=device,
        checkpoint_dir=config.output.checkpoints_dir,
        best_name="hd_gcn_best.pt",
    )
    model = build_hd_gcn(
        num_joints=config.hd_gcn.num_joints,
        hidden_dim=config.hd_gcn.hidden_dim,
        num_classes=config.hd_gcn.num_classes,
        dropout=config.hd_gcn.dropout,
    )
    model = load_checkpoint(model, best_path)
    loss_fn = nn.CrossEntropyLoss()
    _, metrics = evaluate(model.to(device), test_skel, loss_fn, device)
    return model, metrics, best_path


def run_videomae(config: ExperimentConfig, device: torch.device, loaders):
    _, _, _, train_rgb, val_rgb, test_rgb = loaders
    model = build_videomae(
        model_name_or_path=str(config.videomae.checkpoint.init_weights or "MCG-NJU/videomae-base"),
        num_classes=config.videomae.num_classes,
        checkpoint=config.videomae.checkpoint.init_weights,
    )
    best_path = run_training(
        model=model,
        train_loader=train_rgb,
        val_loader=val_rgb,
        epochs=max(1, config.schedule.epochs // 2),
        lr=config.videomae.lr,
        weight_decay=config.schedule.weight_decay,
        device=device,
        checkpoint_dir=config.output.checkpoints_dir,
        best_name="videomae_best.pt",
    )
    model = build_videomae(
        model_name_or_path=str(config.videomae.checkpoint.init_weights or best_path),
        num_classes=config.videomae.num_classes,
        checkpoint=best_path,
    )
    loss_fn = nn.CrossEntropyLoss()
    _, metrics = evaluate(model.to(device), test_rgb, loss_fn, device)
    return model, metrics, best_path


def main(config_path: Path) -> None:
    experiment_set = load_experiment_config(config_path)
    config = experiment_set.base
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loaders = prepare_loaders(config)
    hd_model, hd_metrics, hd_ckpt = run_hd_gcn(config, device, loaders)
    vid_model, vid_metrics, vid_ckpt = run_videomae(config, device, loaders)

    # Save metrics
    metrics_path = config.output.logs_dir / "exp1_metrics.json"
    save_metrics_json(
        {
            "hd_gcn": {
                "accuracy": hd_metrics.accuracy,
                "macro_f1": hd_metrics.macro_f1,
                "checkpoint": str(hd_ckpt),
            },
            "videomae": {
                "accuracy": vid_metrics.accuracy,
                "macro_f1": vid_metrics.macro_f1,
                "checkpoint": str(vid_ckpt),
            },
        },
        metrics_path,
    )

    # Confusion matrices
    # Not storing class names in config, so use indices
    class_names = [f"class_{i}" for i in range(config.hd_gcn.num_classes)]

    hd_confusion_plot = config.output.plots_dir / "exp1_hd_gcn_confusion.png"
    vid_confusion_plot = config.output.plots_dir / "exp1_videomae_confusion.png"
    plot_confusion(class_names=class_names, confusion=hd_metrics.confusion, title="HD-GCN Confusion", output_path=hd_confusion_plot)
    plot_confusion(class_names=class_names, confusion=vid_metrics.confusion, title="VideoMAE Confusion", output_path=vid_confusion_plot)

    print(f"Exp 1 complete. Metrics saved to {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp 1: Modality comparison")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config")
    args = parser.parse_args()
    main(args.config)
