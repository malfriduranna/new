import argparse
from pathlib import Path

import torch

from .datasets import build_loaders, load_manifest
from .engine import inference
from .metrics import classification_metrics
from .models.hd_gcn_adapter import build_hd_gcn
from .models.videomae_adapter import build_videomae
from .utils import load_experiment_config, save_metrics_json, set_seed


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
            model_name_or_path=config.videomae.model_name_or_path,
            num_classes=config.videomae.num_classes,
            checkpoint=vid_ckpt if vid_ckpt.exists() else config.videomae.checkpoint.init_weights,
            eval_only=config.videomae.eval_only,
        )

    skel_preds, skel_labels, skel_logits = inference(hd_model.to(device), test_skel, device)
    rgb_preds = rgb_labels = rgb_logits = None
    if vid_model is not None:
        rgb_preds, rgb_labels, rgb_logits = inference(vid_model.to(device), test_rgb, device)

    if rgb_labels is not None and skel_labels != rgb_labels:
        raise ValueError("Skeleton and RGB test loaders are not aligned; ensure manifests share ordering")

    skel_prob = torch.cat(skel_logits, dim=0)
    labels_tensor = torch.tensor(skel_labels)

    fusion_results = {}
    best_acc = -1.0
    best_alpha = 0.0
    if rgb_logits is not None:
        rgb_prob = torch.cat(rgb_logits, dim=0)
        for alpha in exp_set.fusion.alphas:
            fused = alpha * rgb_prob + (1 - alpha) * skel_prob
            preds = torch.argmax(fused, dim=1)
            metrics = classification_metrics(preds.tolist(), labels_tensor.tolist())
            fusion_results[str(alpha)] = {"accuracy": metrics.accuracy, "macro_f1": metrics.macro_f1}
            if metrics.accuracy > best_acc:
                best_acc = metrics.accuracy
                best_alpha = alpha

    results_path = config.output.logs_dir / "exp5_fusion.json"
    single_modalities = {"hd_gcn": classification_metrics(skel_preds, skel_labels).__dict__}
    if rgb_preds is not None:
        single_modalities["videomae"] = classification_metrics(rgb_preds, rgb_labels).__dict__
    else:
        fusion_results = None
    save_metrics_json(
        {
            "fusion": fusion_results,
            "best": {"alpha": best_alpha, "accuracy": best_acc} if fusion_results else None,
            "single_modalities": single_modalities,
        },
        results_path,
    )

    print(f"Exp 5 complete. Metrics saved to {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp 5: Late fusion")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config")
    args = parser.parse_args()
    main(args.config)
