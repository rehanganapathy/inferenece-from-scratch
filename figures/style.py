"""Shared plotting style. Colors follow the configuration, not its rank:
naive is always blue, KV cache (static) always orange, continuous always aqua.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "bench" / "results"
OUT = Path(__file__).resolve().parent / "out"

COLORS = {
    "naive": "#2a78d6",       # slot 1
    "static": "#eb6834",      # slot 2
    "cached": "#eb6834",      # same entity as static: KV cache, no continuous batching
    "continuous": "#1baf7a",  # slot 3
}
LABELS = {
    "naive": "Naive (no cache)",
    "cached": "KV cache",
    "static": "KV cache, static batching",
    "continuous": "Continuous batching",
}
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT2 = "#52514e"
GRID = "#e6e5e1"


def setup() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": TEXT2, "axes.titlecolor": TEXT,
        "xtick.color": TEXT2, "ytick.color": TEXT2, "text.color": TEXT,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "font.size": 10, "lines.linewidth": 2, "lines.markersize": 6,
        "legend.frameon": False, "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight",
    })


def load(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def end_label(ax, x, y, text, color, dx=6):
    ax.annotate(text, (x, y), xytext=(dx, 0), textcoords="offset points",
                va="center", ha="left", color=color, fontsize=9, fontweight="bold")


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"{name}.{ext}")
    print(f"wrote {OUT / name}.png")
