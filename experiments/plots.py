from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


plt.switch_backend("Agg")


def plot_learning_curve(points: Sequence[Tuple[int, float]], title: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    xs, ys = zip(*sorted(points, key=lambda p: p[0]))
    plt.figure(figsize=(6, 4))
    plt.plot(xs, ys, marker="o")
    plt.xlabel("Training Data Percentage")
    plt.ylabel("Top-1 Accuracy")
    plt.title(title)
    plt.grid(True)
    plt.ylim(0, 1)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_confusion(confusion: List[List[int]], class_names: Sequence[str], title: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.array(confusion)
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=False, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_robustness_curves(degradation_to_accuracy: Dict[str, List[Tuple[float, float]]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    for label, points in degradation_to_accuracy.items():
        xs, ys = zip(*sorted(points, key=lambda p: p[0]))
        plt.plot(xs, ys, marker="o", label=label)
    plt.xlabel("Degradation Level")
    plt.ylabel("Top-1 Accuracy")
    plt.title("Robustness Curves")
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 1)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
