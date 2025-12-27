from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
from torch import nn, optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from .metrics import MetricsResult, classification_metrics, gather_predictions


def setup_optimizer(model: nn.Module, lr: float, weight_decay: float) -> optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    return optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    scaler: GradScaler,
    *,
    amp: bool = True,
) -> float:
    model.train()
    running_loss = 0.0
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=amp):
            logits = model(inputs)
            loss = loss_fn(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += float(loss.detach())
    return running_loss / max(len(dataloader), 1)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    *,
    amp: bool = True,
) -> Tuple[float, MetricsResult]:
    model.eval()
    losses: List[float] = []
    preds: List[int] = []
    labels_all: List[int] = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            with autocast(enabled=amp):
                logits = model(inputs)
                loss = loss_fn(logits, labels)
            losses.append(float(loss.detach()))
            batch_preds, batch_labels = gather_predictions(logits, labels)
            preds.extend(batch_preds)
            labels_all.extend(batch_labels)
    metrics = classification_metrics(preds, labels_all)
    return sum(losses) / max(len(losses), 1), metrics


def save_checkpoint(model: nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_checkpoint(model: nn.Module, path: Path) -> nn.Module:
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)
    return model


def run_training(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    checkpoint_dir: Path,
    best_name: str,
) -> Path:
    optimizer = setup_optimizer(model, lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    scaler = GradScaler()
    model.to(device)

    best_acc = 0.0
    best_path = checkpoint_dir / best_name
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, scaler)
        _, val_metrics = evaluate(model, val_loader, loss_fn, device)
        if val_metrics.accuracy > best_acc:
            best_acc = val_metrics.accuracy
            save_checkpoint(model, best_path)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_acc={val_metrics.accuracy:.4f} macro_f1={val_metrics.macro_f1:.4f}")
    return best_path


def inference(model: nn.Module, dataloader: DataLoader, device: torch.device) -> Tuple[List[int], List[int], List[torch.Tensor]]:
    model.to(device)
    model.eval()
    preds: List[int] = []
    labels_all: List[int] = []
    logits_list: List[torch.Tensor] = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            logits = model(inputs)
            probs = torch.softmax(logits, dim=1)
            batch_preds = torch.argmax(probs, dim=1)
            preds.extend(batch_preds.cpu().tolist())
            labels_all.extend(labels.cpu().tolist())
            logits_list.append(probs.cpu())
    return preds, labels_all, logits_list
