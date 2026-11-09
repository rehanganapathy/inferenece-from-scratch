"""Figure 1: decode tokens/sec versus sequence length, naive versus KV cache.

Inputs: bench/results/seqlen_naive.json, bench/results/seqlen_cached.json
"""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt

from figures.style import COLORS, LABELS, RESULTS, end_label, load, save, setup


def main() -> None:
    setup()
    runs = {m: load(RESULTS / f"seqlen_{m}.json") for m in ("naive", "cached")}
    if not any(runs.values()):
        sys.exit("no seqlen results; run `make seqlen`")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    for mode, run in runs.items():
        if not run:
            continue
        xs = [r["seq_len"] for r in run["results"]]
        tps = [r["decode_tokens_per_s"] for r in run["results"]]
        ms = [1000 * (r["per_token_s"][-1]) for r in run["results"]]
        c = COLORS[mode]
        axes[0].plot(xs, tps, marker="o", color=c)
        end_label(axes[0], xs[-1], tps[-1], LABELS[mode], c)
        axes[1].plot(xs, ms, marker="o", color=c)
        end_label(axes[1], xs[-1], ms[-1], LABELS[mode], c)

    axes[0].set_title("Decode throughput")
    axes[0].set_ylabel("tokens / s")
    axes[0].set_xlabel("prompt length (tokens)")
    axes[0].set_ylim(bottom=0)
    axes[1].set_title("Time per token at the end of the sequence")
    axes[1].set_ylabel("ms / token")
    axes[1].set_xlabel("prompt length (tokens)")
    axes[1].set_ylim(bottom=0)
    for ax in axes:
        ax.margins(x=0.25)
    meta = next(r for r in runs.values() if r)["meta"]
    fig.suptitle(f"Single request, {meta['model']}, {meta.get('gpu', meta['device'])}, {meta['dtype']}",
                 x=0.01, ha="left", fontsize=10, color="#52514e")
    fig.tight_layout()
    save(fig, "fig_naive_vs_seqlen")


if __name__ == "__main__":
    main()
