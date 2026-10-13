"""Shared helpers: seeding, output folders, figure/GIF saving, result tables.

Every lesson writes its artefacts to ``docs/lessonNN/`` and its numbers to
``docs/results/lessonNN.json`` so that ``run.py`` can rebuild the README tables
from real runs.
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
DATA = ROOT / "data"


def seed_everything(seed: int = 0) -> np.random.Generator:
    """Seed Python, NumPy and (if installed) PyTorch; return a NumPy Generator."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:  # lessons 01-03 are pure NumPy
        pass
    return np.random.default_rng(seed)


def set_torch_threads(n: int | None = None) -> None:
    """Use all physical cores; the course is designed for a laptop CPU."""
    import torch

    torch.set_num_threads(n or max(1, os.cpu_count() or 1))


def lesson_dir(lesson: int) -> Path:
    d = DOCS / f"lesson{lesson:02d}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def savefig(fig, lesson: int, name: str, dpi: int = 110) -> Path:
    import matplotlib.pyplot as plt

    path = lesson_dir(lesson) / name
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.relative_to(ROOT)}")
    return path


def save_gif(anim, lesson: int, name: str, fps: int = 10, dpi: int = 70) -> Path:
    """Save a matplotlib FuncAnimation as a (small) GIF with Pillow."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import PillowWriter

    path = lesson_dir(lesson) / name
    anim.save(path, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(anim._fig)
    print(f"  saved {path.relative_to(ROOT)}")
    return path


def save_results(lesson: int, results: dict) -> Path:
    d = DOCS / "results"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"lesson{lesson:02d}.json"
    path.write_text(json.dumps(results, indent=2, default=_to_builtin))
    return path


def load_results(lesson: int) -> dict:
    return json.loads((DOCS / "results" / f"lesson{lesson:02d}.json").read_text())


def _to_builtin(x):
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    raise TypeError(type(x))


def md_table(header: list[str], rows: list[list]) -> str:
    """Tiny Markdown table formatter used for README/results tables."""

    def fmt(v):
        if isinstance(v, float):
            return f"{v:.4g}"
        return str(v)

    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(fmt(v) for v in r) + " |" for r in rows]
    return "\n".join(lines)


def banner(text: str) -> None:
    print("\n" + "=" * 78 + f"\n{text}\n" + "=" * 78)


def setup_matplotlib():
    """Headless backend and a consistent, readable style for all figures."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 100, "axes.grid": True, "grid.alpha": 0.3,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 10, "axes.titlesize": 11, "legend.frameon": False,
        "image.cmap": "viridis", "figure.constrained_layout.use": True,
    })
    return plt


def decision_grid(X, n: int = 200, pad: float = 0.3):
    """A regular grid covering the data, for plotting decision regions."""
    lo, hi = X.min(0) - pad, X.max(0) + pad
    gx, gy = np.meshgrid(np.linspace(lo[0], hi[0], n), np.linspace(lo[1], hi[1], n))
    return gx, gy, np.c_[gx.ravel(), gy.ravel()]
