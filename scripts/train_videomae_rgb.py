import argparse
from pathlib import Path
from typing import Sequence, Tuple

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from experiments.soccernet_rgb import SoccerNetRGBDataset, build_videomae_model


DEFAULT_CLASSES = ["Pass", "Shot", "Ball Touch", "Foul"]


def _make_loaders(
    dataset_root: Path,
    class_names: Sequence[str],
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = SoccerNetRGBDataset(dataset_root, "train", class_names)
    val_ds = SoccerNetRGBDataset(dataset_root, "valid", class_names)
    test_ds = SoccerNetRGBDataset(dataset_root, "test", class_names)

    def loader(ds: SoccerNetRGBDataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )

    return loader(train_ds, True), loader(val_ds, False), loader(test_ds, False)


def _forward_pass(model: nn.Module, batch: torch.Tensor) -> torch.Tensor:
    pixel_values = batch.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W)
    outputs = model(pixel_values=pixel_values)
    return outputs.logits


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: AdamW,
    loss_fn: nn.Module,
    device: torch.device,
    scaler: GradScaler,
) -> float:
    model.train()
    running_loss = 0.0
    for videos, labels in loader:
        videos = videos.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast():
            logits = _forward_pass(model, videos)
            loss = loss_fn(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += float(loss.detach())
    return running_loss / max(len(loader), 1)


def evaluate(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, device: torch.device) -> Tuple[float, float, float]:
    model.eval()
    losses = []
    preds = []
    targets = []
    with torch.no_grad():
        for videos, labels in loader:
            videos = videos.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with autocast():
                logits = _forward_pass(model, videos)
                loss = loss_fn(logits, labels)
            losses.append(float(loss.detach()))
            preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
            targets.extend(labels.cpu().tolist())
    acc = accuracy_score(targets, preds) if targets else 0.0
    macro_f1 = f1_score(targets, preds, average="macro") if targets else 0.0
    return sum(losses) / max(len(losses), 1), acc, macro_f1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train VideoMAE on SoccerNet RGB clips.")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Path to curated dataset root.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/videomae_rgb"), help="Checkpoint output directory.")
    parser.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES, help="List of class names.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader = _make_loaders(args.dataset_root, args.classes, args.batch_size, args.num_workers)
    model = build_videomae_model(args.classes, device=device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scaler = GradScaler()

    best_acc = 0.0
    best_path = args.output_dir / "checkpoints" / "best_videomae.pt"
    best_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, scaler)
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, loss_fn, device)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_path)
        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_macro_f1={val_f1:.4f}"
        )

    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))
    test_loss, test_acc, test_f1 = evaluate(model, test_loader, loss_fn, device)
    print(f"Test: loss={test_loss:.4f} acc={test_acc:.4f} macro_f1={test_f1:.4f}")


if __name__ == "__main__":
    main()
