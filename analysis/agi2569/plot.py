"""
Fig 4 and Fig 5 style plots for AGI-2569, one curve per arm.

    python analysis/agi2569/plot.py <local_records dir> [--out <dir>] [--definition min_any_step|min_final]
                                    [--stop fixed_steps_500] [--compute flops|tokens]

fig4_relearning.png  Attacker's envelope (min over LRs at each step) of Korean loss during relearning, for the
                     reference models and each arm at alpha = 0 (arm C at alpha 0 = "unlearn, then fine-tune on
                     English", the extra curve the plan asks for).
fig5_robustness.png  Normalised robustness vs compute ratio to data filtering, one line per arm over alpha.
                     Fixed-step runs all cost the same, so for them (the default --stop) the x-axis is alpha;
                     pass --stop early_stop for the paper's robustness-vs-compute view.
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aggregate import aggregate, worst_case_curve  # noqa: E402

# Fixed categorical order (dataviz reference palette); an arm keeps its colour in every figure.
ARM_STYLE = {
    "A": ("#2a78d6", "o"), "B": ("#eb6834", "s"), "C": ("#1baf7a", "^"), "D_uniform": ("#eda100", "D"),
    "D_unigram": ("#e87ba4", "v"), "E_uniform": ("#008300", "P"), "E_unigram": ("#4a3aa7", "X"), "F": ("#e34948", "*"),
}
REF_STYLE = {  # references in ink tones, distinguished by dash and label
    "base": ("#0b0b0b", ":", "pretrained base"), "unlearn": ("#52514e", "--", "MaxEnt unlearned"),
    "full_distill": ("#52514e", "-.", "full distill (unlearn -> random init)"), "filter": ("#0b0b0b", "-", "data filtered"),
}


def style_axes(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)


def fig4(curves, out):
    envelope = worst_case_curve(curves)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for role, (color, dash, label) in REF_STYLE.items():
        sub = envelope[envelope.role == role].sort_values("step")
        if len(sub):
            ax.plot(sub.step, sub.kor_loss, color=color, linestyle=dash, linewidth=2, label=label)
    for arm, (color, marker) in ARM_STYLE.items():
        sub = envelope[(envelope.arm == arm) & (envelope.alpha == 0.0)].sort_values("step")
        if len(sub):
            ax.plot(sub.step, sub.kor_loss, color=color, linewidth=2, marker=marker, markevery=10, markersize=6,
                    label=f"arm {arm}, alpha 0")
    style_axes(ax)
    ax.set_xlabel("relearning step")
    ax.set_ylabel("Korean val loss (min over attacker LRs)")
    ax.set_title("Relearning Korean (lower = relearned)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig4_relearning.png"), dpi=160)
    plt.close(fig)


def fig5(summary, out, definition, stop, compute):
    data = summary[summary.definition == definition]
    by_alpha = stop != "early_stop"
    x_col = "alpha" if by_alpha else f"compute_ratio_{compute}"
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.axhline(0, color=REF_STYLE["unlearn"][0], linestyle="--", linewidth=1.5, label="MaxEnt unlearned (0)")
    ax.axhline(1, color=REF_STYLE["filter"][0], linestyle="-", linewidth=1.5, label="data filtered (1)")
    full = data[data.role == "full_distill"]
    if len(full) and not by_alpha:
        ax.scatter(full[x_col], full.robustness, color="#0b0b0b", marker="*", s=120, zorder=3, label="full distill")
    for arm, (color, marker) in ARM_STYLE.items():
        sub = data[(data.arm == arm) & (data.stop == stop)].sort_values("alpha")
        if len(sub) == 0:
            continue
        ax.plot(sub[x_col], sub.robustness, color=color, marker=marker, markersize=7, linewidth=2,
                markeredgecolor="white", markeredgewidth=1, label=f"arm {arm}")
        if not by_alpha:
            last = sub.iloc[-1]
            ax.annotate(f"a={last.alpha:g}", (last[x_col], last.robustness), textcoords="offset points",
                        xytext=(4, 4), fontsize=7, color="#52514e")
    style_axes(ax)
    ax.set_xlabel("noise alpha" if by_alpha else f"compute / data-filtering compute ({compute})")
    ax.set_ylabel("normalised robustness")
    ax.set_title(f"Robustness vs {'alpha' if by_alpha else 'compute'} ({stop}; worst case = {definition})")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig5_robustness.png"), dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("records_root")
    parser.add_argument("--out", default="analysis/agi2569/out")
    parser.add_argument("--definition", choices=["min_any_step", "min_final"], default="min_any_step")
    parser.add_argument("--stop", default="fixed_steps_500")
    parser.add_argument("--compute", choices=["flops", "tokens"], default="flops")
    parser.add_argument("--filter_tokens", type=float, default=None)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    curves, summary = aggregate(args.records_root, args.filter_tokens)
    fig4(curves, args.out)
    fig5(summary, args.out, args.definition, args.stop, args.compute)
    print(f"Wrote {args.out}/fig4_relearning.png and fig5_robustness.png")


if __name__ == "__main__":
    main()
