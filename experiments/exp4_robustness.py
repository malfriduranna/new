import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from .datasets import RGBDataset, SkeletonDataset, load_manifest
from .engine import inference
from .metrics import classification_metrics
from .models.hd_gcn_adapter import build_hd_gcn
from .models.videomae_adapter import build_videomae
from .plots import plot_robustness_curves
from .utils import load_experiment_config, save_metrics_json, set_seed


def build_loader(dataset, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)


def main(config_path: Path) -> None:
    exp_set = load_experiment_config(config_path)
    config = exp_set.base
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_entries = load_manifest(config.dataset.test_manifest, config.dataset.video_root, config.dataset.skeleton_root)

    hd_ckpt = config.output.checkpoints_dir / "hd_gcn_best.pt"
    vid_ckpt = config.output.checkpoints_dir / "videomae_best.pt"

    hd_model = build_hd_gcn(
        num_joints=config.hd_gcn.num_joints,
        hidden_dim=config.hd_gcn.hidden_dim,
        num_classes=config.hd_gcn.num_classes,
        dropout=config.hd_gcn.dropout,
        checkpoint=hd_ckpt if hd_ckpt.exists() else config.hd_gcn.checkpoint.init_weights,
    ).to(device)

    vid_model = None
    if config.videomae.enabled:
        vid_model = build_videomae(
            model_name_or_path=str(config.videomae.checkpoint.init_weights or "MCG-NJU/videomae-base"),
            num_classes=config.videomae.num_classes,
            checkpoint=vid_ckpt if vid_ckpt.exists() else config.videomae.checkpoint.init_weights,
        ).to(device)

    hd_results: List[Tuple[float, float]] = []
    for sigma in exp_set.robustness_noise.sigmas:
        test_dataset = SkeletonDataset(test_entries, sequence_length=config.hd_gcn.sequence_length, noise_sigma=sigma)
        loader = build_loader(test_dataset, config.schedule.batch_size, config.schedule.num_workers)
        preds, labels, _ = inference(hd_model, loader, device)
        metrics = classification_metrics(preds, labels)
        hd_results.append((sigma, metrics.accuracy))

    vid_results: List[Tuple[float, float]] = []
    if vid_model is not None:
        for res in exp_set.robustness_resolution.resolutions:
            test_dataset = RGBDataset(
                test_entries,
                num_frames=config.videomae.num_frames,
                resize_to=config.videomae.image_size,
                center_crop=True,
                downsample_to=res,
            )
            loader = build_loader(test_dataset, config.schedule.batch_size, config.schedule.num_workers)
            preds, labels, _ = inference(vid_model, loader, device)
            metrics = classification_metrics(preds, labels)
            vid_results.append((res, metrics.accuracy))

    metrics_path = config.output.logs_dir / "exp4_robustness.json"
    save_metrics_json(
        {
            "skeleton_noise": {str(k): v for k, v in hd_results},
            "rgb_resolution": {str(k): v for k, v in vid_results} if vid_results else {},
        },
        metrics_path,
    )

    curves = {"HD-GCN noise sigma": hd_results}
    if vid_results:
        curves["VideoMAE resolution"] = vid_results
    plot_robustness_curves(curves, output_path=config.output.plots_dir / "exp4_robustness.png")
    print(f"Exp 4 complete. Metrics saved to {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp 4: Robustness analysis")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config")
    args = parser.parse_args()
    main(args.config)
