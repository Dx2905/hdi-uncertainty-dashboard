
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve

def plot_roc_pr(y_true, proba):
    fpr, tpr, _ = roc_curve(y_true, proba)
    roc_auc = auc(fpr, tpr)

    prec, rec, _ = precision_recall_curve(y_true, proba)
    pr_auc = auc(rec, prec)

    # ROC
    fig1 = plt.figure(figsize=(4.5, 3.6))
    ax1 = fig1.add_subplot(111)
    ax1.plot(fpr, tpr, linewidth=2)
    ax1.plot([0,1], [0,1], linestyle="--", alpha=0.4)
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title(f"ROC (AUC={roc_auc:.3f})")
    ax1.grid(True, alpha=0.2)

    # PR
    fig2 = plt.figure(figsize=(4.5, 3.6))
    ax2 = fig2.add_subplot(111)
    ax2.plot(rec, prec, linewidth=2)
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title(f"PR (AUC={pr_auc:.3f})")
    ax2.grid(True, alpha=0.2)

    return fig1, fig2

def plot_reliability_from_json(rel_path: Path):
    data = json.loads(Path(rel_path).read_text())
    mean_pred = np.array(data["mean_pred"])
    frac_pos  = np.array(data["frac_pos"])

    fig = plt.figure(figsize=(4.5, 3.6))
    ax = fig.add_subplot(111)
    ax.plot([0,1],[0,1], linestyle="--", alpha=0.5)
    ax.plot(mean_pred, frac_pos, linewidth=2, marker="o")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed fraction positive")
    ax.set_title("Reliability diagram (10 bins)")
    ax.grid(True, alpha=0.2)
    return fig
