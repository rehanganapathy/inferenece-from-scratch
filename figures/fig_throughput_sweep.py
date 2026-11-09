"""Figure 3: throughput and TTFT versus concurrency for all three configurations.

Inputs: bench/results/trace_{naive,static,continuous}_c{N}.json
"""

from __future__ import annotations

import re
import sys

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, NullFormatter

from figures.style import COLORS, LABELS, RESULTS, end_label, load, save, setup

MODES = ("naive", "static", "continuous")


def series(mode: str) -> list[tuple[int, dict]]:
    rows = []
    for p in RESULTS.glob(f"trace_{mode}_c*.json"):
        m = re.search(r"_c(\d+)\.json$", p.name)
        if m:
            rows.append((int(m.group(1)), load(p)["summary"]))
    return sorted(rows)


def main() -> None:
    setup()
    data = {m: series(m) for m in MODES}
    if not any(data.values()):
        sys.exit("no trace results; run `make sweep`")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for mode, rows in data.items():
        if not rows:
            continue
        xs = [c for c, _ in rows]
        c = COLORS[mode]
        tps = [s["output_tokens_per_s"] for _, s in rows]
        ttft = [s["ttft_s"]["p50"] for _, s in rows]
        ttft99 = [s["ttft_s"]["p99"] for _, s in rows]
        tpot = [1000 * s["tpot_s"]["p50"] for _, s in rows]
        axes[0].plot(xs, tps, marker="o", color=c)
        end_label(axes[0], xs[-1], tps[-1], LABELS[mode], c)
        axes[1].plot(xs, ttft, marker="o", color=c)
        axes[1].plot(xs, ttft99, color=c, linewidth=1, linestyle="--", alpha=0.7)
        end_label(axes[1], xs[-1], ttft[-1], LABELS[mode], c)
        axes[2].plot(xs, tpot, marker="o", color=c)
        end_label(axes[2], xs[-1], tpot[-1], LABELS[mode], c)

    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("concurrent requests")
        ax.margins(x=0.3)
    axes[0].set_title("Output throughput")
    axes[0].set_ylabel("tokens / s")
    axes[0].set_ylim(bottom=0)
    axes[1].set_title("Time to first token, p50 (dashed: p99)")
    axes[1].set_ylabel("seconds")
    axes[1].set_yscale("log")
    axes[2].set_title("Time per output token, p50")
    axes[2].set_ylabel("ms")
    axes[2].set_ylim(bottom=0)
    fig.tight_layout()
    save(fig, "fig_throughput_sweep")


if __name__ == "__main__":
    main()
