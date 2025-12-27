from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


@dataclass
class MetricsResult:
    accuracy: float
    macro_f1: float
    confusion: List[List[int]]


def gather_predictions(logits: torch.Tensor, labels: torch.Tensor) -> Tuple[List[int], List[int]]:
    probs = torch.softmax(logits, dim=1)
    preds = torch.argmax(probs, dim=1)
    return preds.cpu().tolist(), labels.cpu().tolist()


def classification_metrics(preds: Sequence[int], labels: Sequence[int]) -> MetricsResult:
    acc = accuracy_score(labels, preds)
    macro = f1_score(labels, preds, average="macro")
    conf = confusion_matrix(labels, preds).tolist()
    return MetricsResult(accuracy=acc, macro_f1=macro, confusion=conf)


def aggregate_subset_metrics(
    preds: Sequence[int],
    labels: Sequence[int],
    label_names: Sequence[str],
    subset: Sequence[str],
) -> Dict[str, float]:
    name_to_idx = {name: idx for idx, name in enumerate(label_names)}
    subset_indices = {name_to_idx[name] for name in subset if name in name_to_idx}
    filtered_preds = []
    filtered_labels = []
    for p, y in zip(preds, labels):
        if y in subset_indices:
            filtered_preds.append(p)
            filtered_labels.append(y)
    if not filtered_labels:
        return {"accuracy": 0.0, "macro_f1": 0.0}
    result = classification_metrics(filtered_preds, filtered_labels)
    return {"accuracy": result.accuracy, "macro_f1": result.macro_f1}
