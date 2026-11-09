"""Figure 2: batch slot occupancy over time, static versus continuous batching.

Inputs: bench/results/trace_static_c{N}.json, bench/results/trace_continuous_c{N}.json
Picks the largest N present in both unless --concurrency is given.
"""

from __future__ import annotations

import argparse
import re
import sys

import matplotlib.pyplot as plt
import numpy as np

from figures.style import COLORS, LABELS, RESULTS, load, save, setup


def available(mode: str) -> dict[int, object]:
    out = {}
    for p in RESULTS.glob(f"trace_{mode}_c*.json"):
        m = re.search(r"_c(\d+)\.json$", p.name)
        if m:
            out[int(m.group(1))] = p
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--concurrency", type=int)
    ap.add_argument("--window", type=float, default=None, help="seconds of the run to show")
    a = ap.parse_args()
    setup()

    st, co = available("static"), available("continuous")
    common = sorted(set(st) & set(co))
    if not common:
        sys.exit("need trace_static_c*.json and trace_continuous_c*.json at the same concurrency")
    c = a.concurrency or common[-1]
    runs = {"static": load(st[c]), "continuous": load(co[c])}

    fig, axes = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True)
    tmax = 0
    for ax, (mode, run) in zip(axes, runs.items()):
        steps = [s for s in run["steps"] if s["running"] or s["admitted"]]
        t = np.array([s["t"] for s in steps])
        occ = np.array([s["running"] for s in steps])
        slots = steps[0]["max_slots"]
        tmax = max(tmax, t[-1])
        ax.fill_between(t, occ, step="post", color=COLORS[mode], alpha=0.18, linewidth=0)
        ax.step(t, occ, where="post", color=COLORS[mode], linewidth=1.5)
        ax.axhline(slots, color="#52514e", linewidth=0.8, linestyle=":")
        mean_occ = occ.mean() / slots
        ax.set_title(f"{LABELS[mode]}   mean occupancy {mean_occ:.0%}, "
                     f"{run['summary']['output_tokens_per_s']:.0f} tok/s")
        ax.set_ylabel("busy slots")
        ax.set_ylim(0, slots * 1.12)
        ax.set_yticks([0, slots // 2, slots])
    axes[-1].set_xlabel("time (s)")
    if a.window:
        axes[-1].set_xlim(0, a.window)
    fig.suptitle(f"{c} slots, same trace, {runs['static']['meta'].get('gpu', runs['static']['meta']['device'])}",
                 x=0.01, ha="left", fontsize=10, color="#52514e")
    fig.tight_layout()
    save(fig, "fig_batch_occupancy")


if __name__ == "__main__":
    main()
