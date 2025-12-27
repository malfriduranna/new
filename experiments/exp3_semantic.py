import argparse
from pathlib import Path
from typing import List

import torch

from .datasets import build_loaders, load_manifest
from .engine import inference, load_checkpoint
from .metrics import aggregate_subset_metrics, classification_metrics
from .models.hd_gcn_adapter import build_hd_gcn
from .models.videomae_adapter import build_videomae
from .plots import plot_confusion
from .utils import load_experiment_config, save_metrics_json, set_seed


def load_class_names(config) -> List[str]:
    if config.label_names_path is None:
        return [f"class_{i}" for i in range(config.hd_gcn.num_classes)]
    data = config.label_names_path.read_text().splitlines()
    cleaned = [line.strip() for line in data if line.strip()]
    if len(cleaned) != config.hd_gcn.num_classes:
        raise ValueError("Number of label names does not match num_classes")
    return cleaned


def main(config_path: Path) -> None:
    exp_set = load_experiment_config(config_path)
    config = exp_set.base
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_entries = load_manifest(config.dataset.train_manifest, config.dataset.video_root, config.dataset.skeleton_root)
    val_entries = load_manifest(config.dataset.val_manifest, config.dataset.video_root, config.dataset.skeleton_root)
    test_entries = load_manifest(config.dataset.test_manifest, config.dataset.video_root, config.dataset.skeleton_root)

    loaders = build_loaders(
        train_entries,
        val_entries,
        test_entries,
        sequence_length=config.hd_gcn.sequence_length,
        num_frames=config.videomae.num_frames,
        resize_to=config.videomae.image_size,
        batch_size=config.schedule.batch_size,
        num_workers=config.schedule.num_workers,
    )
    test_skel = loaders[2]
    test_rgb = loaders[5]

    hd_ckpt = config.output.checkpoints_dir / "hd_gcn_best.pt"
    vid_ckpt = config.output.checkpoints_dir / "videomae_best.pt"

    hd_model = build_hd_gcn(
        num_joints=config.hd_gcn.num_joints,
        hidden_dim=config.hd_gcn.hidden_dim,
        num_classes=config.hd_gcn.num_classes,
        dropout=config.hd_gcn.dropout,
        checkpoint=hd_ckpt if hd_ckpt.exists() else config.hd_gcn.checkpoint.init_weights,
    )
    vid_model = None
    if config.videomae.enabled:
        vid_model = build_videomae(
            model_name_or_path=str(config.videomae.checkpoint.init_weights or "MCG-NJU/videomae-base"),
            num_classes=config.videomae.num_classes,
            checkpoint=vid_ckpt if vid_ckpt.exists() else config.videomae.checkpoint.init_weights,
        )

    hd_preds, hd_labels, _ = inference(hd_model.to(device), test_skel, device)
    class_names = load_class_names(config)
    hd_metrics = classification_metrics(hd_preds, hd_labels)
    hd_ball = aggregate_subset_metrics(hd_preds, hd_labels, class_names, exp_set.semantic_groups.ball_interaction)
    hd_body = aggregate_subset_metrics(hd_preds, hd_labels, class_names, exp_set.semantic_groups.body_motion)

    vid_metrics = None
    vid_ball = vid_body = None
    if vid_model is not None:
        vid_preds, vid_labels, _ = inference(vid_model.to(device), test_rgb, device)
        vid_metrics = classification_metrics(vid_preds, vid_labels)
        vid_ball = aggregate_subset_metrics(vid_preds, vid_labels, class_names, exp_set.semantic_groups.ball_interaction)
        vid_body = aggregate_subset_metrics(vid_preds, vid_labels, class_names, exp_set.semantic_groups.body_motion)

    results_path = config.output.logs_dir / "exp3_semantic.json"
    payload = {
        "hd_gcn": {
            "overall": hd_metrics.__dict__,
            "ball_interaction": hd_ball,
            "body_motion": hd_body,
        }
    }
    if vid_metrics is not None and vid_ball is not None and vid_body is not None:
        payload["videomae"] = {
            "overall": vid_metrics.__dict__,
            "ball_interaction": vid_ball,
            "body_motion": vid_body,
        }
    save_metrics_json(payload, results_path)

    plot_confusion(hd_metrics.confusion, class_names, "HD-GCN Semantic Confusion", config.output.plots_dir / "exp3_hd_confusion.png")
    if vid_metrics is not None:
        plot_confusion(vid_metrics.confusion, class_names, "VideoMAE Semantic Confusion", config.output.plots_dir / "exp3_videomae_confusion.png")

    print(f"Exp 3 complete. Metrics saved to {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp 3: Semantic grouping analysis")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config")
    args = parser.parse_args()
    main(args.config)
