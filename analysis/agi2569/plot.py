"""
Fig 4 and Fig 5 style plots for AGI-2569, one curve per arm.

    python analysis/agi2569/plot.py <local_records dir> [--out <dir>] [--definition min_final|min_any_step]
                                    [--stop early_stop|fixed_steps_500] [--compute steps|flops|tokens] [--paper]
                                    [--teacher_lr 8e-5]

fig4_relearning.png  Attacker's envelope (min over LRs at each step) of Korean loss during relearning, for the
                     reference models and each arm at alpha = 0 (arm C at alpha 0 = "unlearn, then fine-tune on
                     English", the extra curve the plan asks for).
fig5_robustness.png  Normalised robustness vs compute ratio to data filtering, one line per arm over alpha.
                     Early-stop runs (the default --stop, as in Fig 5a) use the compute ratio as x-axis.
                     Fixed-step runs all cost the same, so with --stop fixed_steps_500 the x-axis is alpha.
fig5b_relearning.png Attacker's envelope per alpha for arm A (--fig5b_arm) and the references (Fig 5b).

--paper overlays the paper's language values (paper_targets.json; see paper_targets.md) as thin dashed lines
and hollow markers. The paper's robustness is the `min_final` definition, the default here.
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aggregate import TEACHER_LR, aggregate, worst_case_curve  # noqa: E402

PAPER_TARGETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper_targets.json")
PAPER_KW = dict(linestyle=(0, (4, 2)), linewidth=1.2, alpha=0.6)  # paper overlays: thin, dashed
FIG5B_ALPHAS = (0.2, 0.4, 0.6, 0.8)
ALPHA_SHADES = {0.2: "#9ec5f0", 0.4: "#5598e0", 0.6: "#2a6fc4", 0.8: "#173f78"}  # light -> dark with alpha


def load_paper(path=PAPER_TARGETS):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def plot_paper_curve(ax, points, color, label):
    steps, losses = zip(*points)
    ax.plot(steps, losses, color=color, label=label, **PAPER_KW)

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


def fig4(curves, out, paper=None):
    envelope = worst_case_curve(curves)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for role, (color, dash, label) in REF_STYLE.items():
        sub = envelope[envelope.role == role].sort_values("step")
        if len(sub):
            ax.plot(sub.step, sub.kor_loss, color=color, linestyle=dash, linewidth=2, label=label)
    if paper:
        for role, color in (("unlearn", "#a35bb0"), ("full_distill", "#2a78d6"), ("filter", "#c27c00")):
            plot_paper_curve(ax, paper["fig4b_maxent_language"]["curves"][role], color, f"paper: {REF_STYLE[role][2]}")
        ax.set_ylim(top=7.5)
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


def fig5(summary, out, definition, stop, compute, paper=None):
    data = summary[summary.definition == definition]
    by_alpha = stop != "early_stop"
    x_col = "alpha" if by_alpha else f"compute_ratio_{compute}"
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if paper:
        pts = paper["fig5a_language"]["points"]
        xs = [p["alpha"] if by_alpha else p["compute_pct"] / 100 for p in pts]
        ys = [p["robustness_pct"] / 100 for p in pts]
        yerr = [[(p["robustness_pct"] - p["robustness_pct_bar"][0]) / 100 for p in pts],
                [(p["robustness_pct_bar"][1] - p["robustness_pct"]) / 100 for p in pts]]
        ax.errorbar(xs, ys, yerr=yerr, fmt="o", mfc="none", mec="#52514e", ecolor="#a3a29c", markersize=7,
                    capsize=2, label="paper UNDO (early stop, median of 5 seeds)")
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
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig5_robustness.png"), dpi=160)
    plt.close(fig)


def fig5b(curves, out, arm, stop, paper=None):
    """Fig 5b: attacker's envelope for one arm at alpha 0.2/0.4/0.6/0.8, with the unlearned and filtered refs."""
    envelope = worst_case_curve(curves)
    stops = curves.drop_duplicates("model").set_index("model").stop
    envelope = envelope[envelope.model.map(stops).isin([stop]) | envelope.role.isin(["unlearn", "filter"])]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for role in ("unlearn", "filter"):
        color, dash, label = REF_STYLE[role]
        sub = envelope[envelope.role == role].sort_values("step")
        if len(sub):
            ax.plot(sub.step, sub.kor_loss, color=color, linestyle=dash, linewidth=2, label=label)
    for alpha in FIG5B_ALPHAS:
        sub = envelope[(envelope.arm == arm) & (envelope.alpha.round(6) == alpha)].sort_values("step")
        if len(sub):
            ax.plot(sub.step, sub.kor_loss, color=ALPHA_SHADES[alpha], linewidth=2, label=f"arm {arm}, alpha {alpha:g}")
    if paper:
        curves5 = paper["fig5b_language"]["curves"]
        plot_paper_curve(ax, curves5["unlearn"], "#52514e", "paper: MaxEnt unlearned")
        plot_paper_curve(ax, curves5["filter"], "#0b0b0b", "paper: data filtered")
        for alpha in FIG5B_ALPHAS:
            plot_paper_curve(ax, curves5[f"alpha_{alpha:g}"], ALPHA_SHADES[alpha], f"paper: UNDO alpha {alpha:g}")
        ax.set_ylim(top=7.5)
    style_axes(ax)
    ax.set_xlabel("relearning step")
    ax.set_ylabel("Korean val loss (min over attacker LRs)")
    ax.set_title(f"Relearning Korean after arm {arm} ({stop})")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig5b_relearning.png"), dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("records_root")
    parser.add_argument("--out", default="analysis/agi2569/out")
    parser.add_argument("--definition", choices=["min_final", "min_any_step"], default="min_final")
    parser.add_argument("--stop", default="early_stop")
    parser.add_argument("--teacher_lr", type=float, default=TEACHER_LR, help="MaxEnt teacher LR (default 8e-5)")
    parser.add_argument("--compute", choices=["steps", "flops", "tokens"], default="steps")
    parser.add_argument("--filter_tokens", type=float, default=None)
    parser.add_argument("--filter_steps", type=float, default=None)
    parser.add_argument("--fig5b_arm", default="A")
    parser.add_argument("--paper", action="store_true", help="overlay the paper's values (paper_targets.json)")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    curves, summary = aggregate(args.records_root, args.filter_tokens, args.filter_steps, args.teacher_lr)
    paper = load_paper() if args.paper else None
    fig4(curves, args.out, paper)
    fig5(summary, args.out, args.definition, args.stop, args.compute, paper)
    fig5b(curves, args.out, args.fig5b_arm, args.stop, paper)
    print(f"Wrote {args.out}/fig4_relearning.png, fig5_robustness.png and fig5b_relearning.png")


if __name__ == "__main__":
    main()
