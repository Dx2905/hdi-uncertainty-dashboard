
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

def plot_interval_band(samples: np.ndarray):
    """
    samples: 1D array of bootstrap probabilities (0..1)
    Returns: matplotlib Figure
    """
    s = np.asarray(samples).ravel()
    q025, med, q975 = np.quantile(s, [0.025, 0.5, 0.975])

    fig = plt.figure(figsize=(5, 1.6))
    ax = fig.add_subplot(111)
    ax.hlines(1, q025, q975, linewidth=8, alpha=0.25)
    ax.vlines(med, 0.8, 1.2, linewidth=3)
    ax.set_ylim(0.7, 1.3)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Predicted risk")
    ax.set_yticks([])
    ax.grid(True, axis="x", alpha=0.2)
    ax.set_title("Encoding A: Median + 95% interval")
    return fig

def plot_quantile_dotplot(samples: np.ndarray, n_dots: int = 50):
    """
    samples: 1D array of bootstrap probabilities (0..1)
    n_dots: number of quantile dots to show
    Returns: matplotlib Figure
    """
    s = np.asarray(samples).ravel()
    qs = np.linspace(0, 1, n_dots)
    dots = np.quantile(s, qs)

    fig = plt.figure(figsize=(5, 1.6))
    ax = fig.add_subplot(111)
    y = np.ones_like(dots)
    ax.scatter(dots, y, s=25, alpha=0.9)
    ax.set_ylim(0.7, 1.3)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Predicted risk")
    ax.set_yticks([])
    ax.grid(True, axis="x", alpha=0.2)
    ax.set_title("Encoding B: Quantile dotplot")
    return fig
