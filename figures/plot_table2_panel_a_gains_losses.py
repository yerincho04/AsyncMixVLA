#!/usr/bin/env python3
"""Plot Table 2 outcome gains/losses directly from the frozen final report."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPORT = (
    ROOT
    / "results"
    / "table2_learned_k4_jointv2"
    / "v1_v2_cascade_continuous_TEST_K4_JOINTV2.json"
)
OUT = ROOT / "figures" / "table2_panel_a_gains_losses"

METHODS = [
    ("adapter", "Adapter only"),
    ("full_oft", "Full OFT only"),
    ("sync", "Synchronous switching"),
    ("naive", "Naive asynchronous"),
    ("ours", "AsyncMixVLA (ours)"),
]


def main() -> None:
    report = json.loads(REPORT.read_text())
    if report.get("matched") != report.get("expected") or report.get("integrity"):
        raise RuntimeError("Refusing to plot an incomplete or invalid Table 2 report")

    rows = report["rows"]
    n = int(report["matched"])
    labels, gains, losses, success_rates = [], [], [], []
    for key, label in METHODS:
        row = rows[key]
        labels.append(label)
        gains.append(int(row["rescued"]))
        losses.append(int(row["retain_n"]) - int(row["retained"]))
        success_rates.append(100.0 * float(row["overall_sr"]))

    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(12.0, 5.8), dpi=170)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Highlight the proposed method across the complete row.
    ours_index = len(labels) - 1
    ax.axhspan(ours_index - 0.5, ours_index + 0.5, color="#f1f3f5", zorder=0)

    loss_colors = ["#9aa3ad"] + ["#df9b8e"] * 3 + ["#c6533e"]
    gain_colors = ["#9aa3ad"] + ["#83b9a7"] * 3 + ["#31836a"]
    ax.barh(y, -np.asarray(losses), height=0.56, color=loss_colors, edgecolor="none", zorder=3)
    ax.barh(y, gains, height=0.56, color=gain_colors, edgecolor="none", zorder=3)

    ax.axvline(0, color="#252a30", linewidth=1.4, zorder=4)
    ax.set_xlim(-30, 30)
    ax.set_ylim(len(labels) - 0.45, -0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=15, color="#24292f")
    ax.tick_params(axis="y", length=0, pad=20)
    for tick, (key, _) in zip(ax.get_yticklabels(), METHODS):
        if key == "ours":
            tick.set_fontweight("bold")

    ticks = np.arange(-30, 31, 10)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(abs(int(v))) for v in ticks], fontsize=12, color="#77808c")
    ax.set_xlabel(f"Episodes (N = {n})", fontsize=14, color="#77808c", labelpad=8)
    ax.grid(axis="x", color="#d7dce2", linewidth=0.8, zorder=1)

    for i, (key, _) in enumerate(METHODS):
        if key == "adapter":
            ax.scatter(0, i, marker="D", s=65, color="#89939f", zorder=5)
            ax.text(1.15, i, "reference", va="center", ha="left", fontsize=13,
                    color="#7d8793", style="italic")
            continue
        left_weight = "bold" if key == "ours" else "normal"
        right_weight = "bold" if key == "ours" else "normal"
        ax.text(-losses[i] - 0.7, i, f"−{losses[i]}", va="center", ha="right",
                fontsize=14, color="#c94c39", fontweight=left_weight)
        ax.text(gains[i] + 0.7, i, f"+{gains[i]}", va="center", ha="left",
                fontsize=14, color="#247c65", fontweight=right_weight)

    # Right-hand success-rate column, positioned outside the bar axes.
    for i, ((key, _), sr) in enumerate(zip(METHODS, success_rates)):
        ax.text(1.115, i, f"{sr:.2f}%", transform=ax.get_yaxis_transform(),
                va="center", ha="center", fontsize=15, color="#24292f",
                fontweight="bold" if key == "ours" else "normal", clip_on=False)

    ax.text(0.5, 1.105, "relative to Adapter only", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=13, color="#7d8793", style="italic")
    ax.text(0.49, 1.055, "← Successes lost", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=14, color="#c94c39")
    ax.text(0.51, 1.055, "Failures rescued →", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=14, color="#247c65")
    ax.text(1.115, 1.055, "Overall SR", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=14, color="#747e8a", clip_on=False)

    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#d7dce2")
    plt.subplots_adjust(left=0.31, right=0.80, top=0.76, bottom=0.19)

    png_metadata = {
        "Title": "Table 2 panel A: gains and losses",
        "Description": f"Generated from {REPORT.relative_to(ROOT)}; matched={n}; k=4 joint-v2",
    }
    pdf_metadata = {
        "Title": "Table 2 panel A: gains and losses",
        "Subject": f"Generated from {REPORT.relative_to(ROOT)}; matched={n}; k=4 joint-v2",
    }
    fig.savefig(OUT.with_suffix(".png"), bbox_inches="tight", facecolor="white", metadata=png_metadata)
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight", facecolor="white", metadata=pdf_metadata)
    print(f"source: {REPORT}")
    for (key, label), gain, loss, sr in zip(METHODS, gains, losses, success_rates):
        print(f"{key:8s} {label:26s} loss={loss:2d} gain={gain:2d} sr={sr:.2f}%")
    print(f"wrote {OUT.with_suffix('.png')}")
    print(f"wrote {OUT.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
