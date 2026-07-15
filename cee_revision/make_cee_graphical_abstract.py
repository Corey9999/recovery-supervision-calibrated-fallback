"""Create the optional CEE graphical abstract from the frozen fact ledger."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "figures" / "graphical_abstract_cee"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
)

BLUE = "#2F74B5"
TEAL = "#2A9D8F"
ORANGE = "#E07A2D"
GRAY = "#5E6673"
LIGHT_BLUE = "#EAF2FA"
LIGHT_TEAL = "#EAF7F3"
LIGHT_ORANGE = "#FFF2E7"


def box(ax, x, y, w, h, text, edge, face, size=11, weight="normal"):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        linewidth=1.6,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, fontweight=weight, color="#20242A")


def arrow(ax, x1, y1, x2, y2, color=GRAY):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14, linewidth=1.6, color=color))


def metric_card(ax, x, y, value, label, edge, face):
    w, h = 0.102, 0.17
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.006,rounding_size=0.012",
        linewidth=1.6,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + 0.112, value, ha="center", va="center", fontsize=11.5, fontweight="bold", color="#20242A")
    ax.text(x + w / 2, y + 0.055, label, ha="center", va="center", fontsize=8.2, fontweight="bold", linespacing=0.95, color="#20242A")


def main():
    fig, ax = plt.subplots(figsize=(13.28, 5.31), dpi=100)
    fig.subplots_adjust(left=0.018, right=0.988, top=0.94, bottom=0.08)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.02, 0.93, "Support-gated selective recovery under controlled sensor corruption", fontsize=18, fontweight="bold", color="#15191F", va="top")
    ax.text(0.02, 0.855, "Endpoints and routing thresholds were frozen before test; the support gate was proposed afterward and audited retrospectively", fontsize=10.5, color=GRAY, va="top")

    box(ax, 0.02, 0.40, 0.145, 0.25, "Controlled fault\nreaches an available\nsensor group", BLUE, LIGHT_BLUE, size=11.5, weight="bold")
    arrow(ax, 0.17, 0.525, 0.215, 0.525)

    box(ax, 0.22, 0.57, 0.13, 0.15, "Base model", GRAY, "#F2F3F5", size=12, weight="bold")
    box(ax, 0.22, 0.32, 0.13, 0.15, "Recovery model", ORANGE, LIGHT_ORANGE, size=12, weight="bold")
    arrow(ax, 0.165, 0.525, 0.215, 0.645, BLUE)
    arrow(ax, 0.165, 0.525, 0.215, 0.395, ORANGE)

    box(ax, 0.405, 0.39, 0.16, 0.27, "Cross-fitted\nconditional selector\n+ post hoc\nsupport-gate audit", TEAL, LIGHT_TEAL, size=11.5, weight="bold")
    arrow(ax, 0.35, 0.645, 0.40, 0.585, BLUE)
    arrow(ax, 0.35, 0.395, 0.40, 0.465, ORANGE)
    arrow(ax, 0.57, 0.525, 0.615, 0.525, TEAL)

    box(ax, 0.62, 0.40, 0.13, 0.25, "Retrospective audit:\nfallback unless\nsupport passes", TEAL, LIGHT_TEAL, size=11.5, weight="bold")

    cards = [
        (0.775, 0.63, "96.2%", "harmful transfers\nprevented", TEAL, LIGHT_TEAL),
        (0.775, 0.40, "6.6%", "available corrections\nretained", ORANGE, LIGHT_ORANGE),
        (0.890, 0.63, "5/10", "fitted pairs\npassed gate", BLUE, LIGHT_BLUE),
        (0.890, 0.40, "2 passes", "supported-pair\nendpoint cost", GRAY, "#F2F3F5"),
    ]
    for x, y, value, label, edge, face in cards:
        metric_card(ax, x, y, value, label, edge, face)

    ax.text(
        0.02,
        0.18,
        "Analytical controls at 2:1: two-stage mean utility +37.8 per 10,000 (95% fitted-pair interval -2.7 to +78.3); not a deployment recommendation.",
        fontsize=9.7,
        color="#353B44",
    )
    ax.text(
        0.02,
        0.09,
        "Outcome: reproducible risk control for controlled corruption; natural-failure recovery\n"
        "and device-independent transfer remain unproven.",
        fontsize=10.8,
        fontweight="bold",
        color="#20242A",
    )

    fig.savefig(OUT.with_suffix(".png"), dpi=100, facecolor="white")
    fig.savefig(OUT.with_suffix(".tiff"), dpi=600, facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(OUT.with_suffix(".pdf"), facecolor="white")
    fig.savefig(OUT.with_suffix(".svg"), facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
