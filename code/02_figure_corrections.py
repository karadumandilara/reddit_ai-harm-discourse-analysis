# === 1. SUMMARY FIGURE: how the estimate changes under successive corrections =========
# Numbers are copied from the outputs recorded in revision_dossier.tex (no data needed).
# None = no confidence interval was computed for that stage.
import matplotlib.pyplot as plt

STAGES = ["Naive verb list", "Validated verbs and objects", "+ subreddit mix fixed",
          "+ topic mix fixed"]
# (before, before_lo, before_hi, after, after_lo, after_hi)
DATA = {
    "All harm verbs": [
        (5.0, None, None, 11.6, None, None),
        (8.2, 6.3, 10.4, 16.9, 15.1, 18.9),
        (9.6, None, None, 15.6, None, None),
        (11.9, 8.9, 15.0, 15.3, 13.6, 16.9)],
    "Without “replace”": [
        (4.1, None, None, 7.1, None, None),
        (7.1, 5.3, 8.9, 10.7, 9.1, 12.3),
        (7.6, None, None, 10.4, None, None),
        (9.0, 6.6, 11.5, 9.4, 7.9, 10.9)],
}
BEFORE, AFTER = "#2a78d6", "#eb6834"                 # validated categorical pair
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": INK})
fig, axes = plt.subplots(1, 2, figsize=(8.2, 2.9), sharey=True)
y = np.arange(len(STAGES))[::-1]
for ax, (title, rows) in zip(axes, DATA.items()):
    for yi, (b, blo, bhi, a, alo, ahi) in zip(y, rows):
        ax.plot([b, a], [yi, yi], color=MUTED, lw=2, zorder=1, solid_capstyle="round")
        if blo is not None:
            ax.plot([blo, bhi], [yi + 0.14] * 2, color=BEFORE, lw=1.2, zorder=2)
            ax.plot([alo, ahi], [yi - 0.14] * 2, color=AFTER, lw=1.2, zorder=2)
        ax.scatter(b, yi, s=46, facecolor="white", edgecolor=BEFORE, linewidth=2, zorder=3)
        ax.scatter(a, yi, s=46, color=AFTER, edgecolor="white", linewidth=1, zorder=3)
        ax.text(24.6, yi, f"{a - b:+.1f}", va="center", ha="right", color=INK, fontsize=8.5)
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.set_xlim(0, 25); ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_xlabel("AI system as agent (% of harm clauses)")
    ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.text(24.6, y[0] + 0.55, "change", ha="right", color=MUTED, fontsize=8)
axes[0].set_yticks(y); axes[0].set_yticklabels(STAGES)
handles = [plt.Line2D([], [], marker="o", ls="", mfc="white", mec=BEFORE, mew=2, ms=7, label="Before ChatGPT"),
           plt.Line2D([], [], marker="o", ls="", color=AFTER, ms=7, label="After ChatGPT")]
fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.58, -0.04))
fig.tight_layout(rect=(0, 0.06, 1, 1))
fig.savefig(f"{OUT}/fig_corrections.pdf", bbox_inches="tight")
fig.savefig(f"{OUT}/fig_corrections.png", dpi=300, bbox_inches="tight")
plt.show()
print("Saved fig_corrections.pdf and .png. Horizontal bars = 95% CI where computed.")
