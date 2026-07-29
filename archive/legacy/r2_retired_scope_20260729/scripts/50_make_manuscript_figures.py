#!/usr/bin/env python3
"""Generate manuscript figures for the R2 sparse-readout audit.

All figures are vector PDF + 600-dpi PNG. Quantitative panels are drawn from
staged result tables so that every plotted value is traceable to an evidence
file; Figure 1 is a vector workflow schematic of the study design.

  Figure 1  Evidential ladder and audit limitations
  Figure 2  Conserved-triplet atlas, characterization, checkpoint sensitivity
  Figure 3  N-terminal readouts with high unnormalized received attention
  Figure 4  Calibrated negatives: enzyme-class steering and causal gates
  Figure 5  Generated lysozyme structural validation
  Figure 6  Representation-recoverability audit (ceiling/floor/rho, EC confound)
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

REPO = Path(__file__).resolve().parents[2]
CA = REPO / "r2_interpretability_transfer/results/circuit_analysis"
SB = REPO / "r2_interpretability_transfer/results/steering_benchmark"
OUT = REPO / "r2_interpretability_transfer/manuscript/figures"
OUT.mkdir(parents=True, exist_ok=True)

# Colour-blind-safe palette (Okabe-Ito).
CB = {
    "blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
    "red": "#D55E00", "purple": "#CC79A7", "sky": "#56B4E9",
    "yellow": "#F0E442", "grey": "#999999", "dark": "#222222",
}
TESTS = ["k-mer", "positional", "bpe-boundary", "attention-sink", "high-norm"]
TEST_DISPLAY = {
    "k-mer": "k-mer",
    "positional": "positional",
    "bpe-boundary": "bpe-boundary",
    "attention-sink": "received attention",
    "high-norm": "high-norm",
}
TEST_COL = {"k-mer": "k_mer_q", "positional": "positional_q",
            "bpe-boundary": "bpe_boundary_q", "attention-sink": "attention_sink_q",
            "high-norm": "high_norm_q"}


def nature_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "axes.titlesize": 7.5,
        "axes.titleweight": "regular",
        "axes.titlepad": 4.0,
        "axes.labelsize": 7.5,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5,
        "axes.linewidth": 0.6,
        "axes.axisbelow": True,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.6,
        "ytick.major.size": 2.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": "#d9d9d9",
        "grid.linewidth": 0.5,
        "legend.handlelength": 1.2,
        "legend.handletextpad": 0.5,
        "legend.borderpad": 0.2,
        "legend.columnspacing": 1.0,
        "figure.facecolor": "white",
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "savefig.facecolor": "white",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "mathtext.default": "regular",
    })


def panel_label(ax, label, dx=-0.085, dy=1.06):
    txt = getattr(ax, "text2D", ax.text)  # 3D axes need text2D for 2D placement
    txt(dx, dy, label, transform=ax.transAxes, fontsize=10,
        fontweight="bold", va="top", ha="right")


def _ygrid(ax):
    ax.grid(axis="y", which="major", zorder=0)
    ax.set_axisbelow(True)


def read_tsv(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def save(fig, name: str) -> None:
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT/name}.pdf")


# --------------------------------------------------------------------------
# Figure 1: evidential ladder and audit limitations
# --------------------------------------------------------------------------

def figure1() -> None:
    """Draw the four independent assessment estimands and their gates."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def box(x, y, w, h, face, edge, text, *, fontsize=6.4,
            weight="regular", color=CB["dark"], radius=0.012, lw=0.8,
            zorder=2):
        patch = FancyBboxPatch(
            (x, y), w, h,
            boxstyle=f"round,pad=0.008,rounding_size={radius}",
            facecolor=face, edgecolor=edge, linewidth=lw, zorder=zorder,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fontsize, fontweight=weight, color=color,
                linespacing=1.25, zorder=zorder + 1)
        return patch

    def arrow(x0, y0, x1, y1, *, color="#6f6f6f", lw=1.0,
              style="-|>", zorder=1):
        ax.add_patch(FancyArrowPatch(
            (x0, y0), (x1, y1), arrowstyle=style, mutation_scale=8,
            color=color, linewidth=lw, shrinkA=1, shrinkB=1, zorder=zorder,
        ))

    ax.text(0.5, 0.985,
            "Four independent estimands: recurrence is not semantics, mechanism or utility",
            ha="center", va="top", fontsize=8.6, fontweight="bold")
    box(
        0.12, 0.89, 0.76, 0.055, "#f5f5f5", "#bdbdbd",
        "ProtGPT2  ·  ZymCTRL  ·  ProGen2-medium   →   architecture-specific CLT inputs and sparse readouts",
        fontsize=6.25,
    )

    cols = [0.015, 0.262, 0.509, 0.756]
    width = 0.229
    header_colors = [CB["blue"], CB["green"], CB["purple"], CB["orange"]]
    headers = [
        "R  ·  RECURRENCE",
        "S  ·  RESIDUAL SEMANTICS",
        "C  ·  CAUSAL COMPUTATION",
        "U  ·  DOWNSTREAM UTILITY",
    ]
    questions = [
        "Do frozen correspondences\nreplicate on independent data?",
        "Does a biological label add\nheld-out conditional information?",
        "Does a faithful intervention alter\ncomputation beyond matched controls?",
        "Does the intervention improve a\nprospectively validated endpoint?",
    ]
    designs = [
        "discover on cohort A\nscore frozen matches on B\ncoherent model-wise null",
        "continuous activations\nprotein/family-blocked folds\ncondition on k-mer · position · norm",
        "disjoint discovery/evaluation\nfeature-fidelity + off-target checks\npositive and matched controls",
        "complete paired generations\nequal-arm selection\nvalidated functional endpoint",
    ]
    gates = [
        "GATE\nheld-out correlation +\nstability across seeds/matchers",
        "GATE\nFDR-controlled residual effect\nor powered small-effect bound",
        "GATE\npositive-control sensitivity +\neffect or equivalence bound",
        "GATE\ngeneration-wide benefit or\nequivalence across all classes",
    ]
    status = [
        "CURRENT\nprocedure/cohort-sensitive;\nconfirmatory run pending",
        "CURRENT\ntop-event matched audit: 0/380;\ncontinuous test pending",
        "CURRENT\nsynthetic control passes; pretrained\nresults remain bounded negatives",
        "CURRENT\nolder mixed/asymmetric run shows\nno validated benefit",
    ]

    for x, hc, hdr, q, design, gate, current in zip(
            cols, header_colors, headers, questions, designs, gates, status):
        box(x, 0.79, width, 0.055, hc, hc, hdr, fontsize=5.7,
            weight="bold", color="white", radius=0.008)
        box(x, 0.68, width, 0.075, "white", "#b7b7b7", q, fontsize=5.35)
        arrow(x + width / 2, 0.677, x + width / 2, 0.645)
        box(x, 0.515, width, 0.12, "#f7f7f7", "#b7b7b7", design, fontsize=5.15)
        arrow(x + width / 2, 0.512, x + width / 2, 0.48)
        box(x, 0.37, width, 0.10, "#eef6fb", hc, gate,
            fontsize=5.0, weight="bold")
        box(x, 0.225, width, 0.105, "#fff3df", CB["orange"], current,
            fontsize=4.85, color="#6a4700")

    box(0.045, 0.045, 0.91, 0.12, "#f8f8f8", "#8f8f8f", "", radius=0.014)
    ax.text(0.065, 0.142, "FOUNDATION REQUIRED BEFORE ANY DOWNSTREAM CLAIM",
            ha="left", va="top", fontsize=6.1, fontweight="bold")
    ax.text(
        0.5, 0.094,
        "mask-aware multi-seed dictionary quality  ·  immutable train/validation/test cohorts  ·  nested recoverability  ·  complete hashes and provenance",
        ha="center", va="center", fontsize=5.35,
    )
    ax.text(
        0.5, 0.058,
        "Each estimand uses independent data and its own acceptance gate; arrows between R, S, C and U are deliberately absent.",
        ha="center", va="center", fontsize=5.15, color="#6a4700",
    )

    save(fig, "figure_01_workflow")


# --------------------------------------------------------------------------
# Figure 2: atlas, characterization, single checkpoint-pair sensitivity
# --------------------------------------------------------------------------

def figure2() -> None:
    null = json.loads((CA / "universal_atlas_balanced200_wide_null_control_30x_20260513.json").read_text())
    obs = null["observed_triplet_counts"]
    runs = null["null_runs"]
    thr = ["0.9", "0.95", "0.98"]
    null_by_thr = {t: np.array([r["triplet_counts"][t] for r in runs], float) for t in thr}

    sig = read_tsv(CA / "triplet_synthesis_20260515_nperm2000/triplet_signatures.tsv")
    char = read_tsv(CA / "triplet_characterization_20260515_nperm2000/triplet_characterization.tsv")
    qmat = {r["triplet_id"]: r for r in char}

    overlap = read_tsv(CA / "triplet_synthesis_20260515_nperm2000/cross_test_overlap.tsv")
    jac = np.zeros((5, 5))
    idx = {t: i for i, t in enumerate(TESTS)}
    for row in overlap:
        jac[idx[row["test_a"]], idx[row["test_b"]]] = float(row["jaccard"])

    qd = json.loads((CA / "universal_atlas_quality_diagnostic_20260516/summary.json").read_text())
    rows = {r["name"]: r for r in qd["atlas_rows"]}

    fig = plt.figure(figsize=(7.2, 6.0))
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.32,
                          height_ratios=[1, 1.15])

    # (a) observed vs permutation null
    ax = fig.add_subplot(gs[0, 0])
    _ygrid(ax)
    x = np.arange(3)
    obs_vals = [obs[t] for t in thr]
    null_means = [null_by_thr[t].mean() for t in thr]
    null_sds = [null_by_thr[t].std() for t in thr]
    ax.bar(x - 0.21, obs_vals, 0.40, color=CB["blue"], label="observed",
           zorder=3, edgecolor="white", linewidth=0.4)
    ax.bar(x + 0.21, null_means, 0.40, yerr=null_sds, color=CB["grey"],
           ecolor=CB["dark"], capsize=2.2, error_kw={"lw": 0.7}, zorder=3,
           edgecolor="white", linewidth=0.4, label="permutation null (30×)")
    for xi, v in zip(x - 0.21, obs_vals):
        ax.text(xi, v + 0.8, str(v), ha="center", va="bottom",
                fontsize=7.5, fontweight="bold", color=CB["blue"])
    for xi, v in zip(x + 0.21, null_means):
        ax.text(xi, v + 1.4, f"{v:.2f}", ha="center", va="bottom",
                fontsize=5.6, color=CB["dark"])
    ax.set_xticks(x)
    ax.set_xticklabels([r"$|r|\geq 0.90$", r"$|r|\geq 0.95$", r"$|r|\geq 0.98$"])
    ax.set_ylabel("matched triplets")
    ax.set_ylim(0, 44)
    ax.legend(frameon=False, loc="upper right", borderaxespad=0.2)
    ax.set_title("Recurrence vs. sequence-assignment null")
    panel_label(ax, "a", dx=-0.11, dy=1.10)

    # (b) signature heatmap: 38 triplets x 5 tests. Only cells significant after
    #     BH correction (q<0.05) are shaded, by -log10(q); the rest are greyed,
    #     so the dominant signatures read out as solid colour blocks per row.
    order = sorted(sig, key=lambda r: (r["cluster"], int(r["rank"])))
    tids = [r["triplet_id"] for r in order]
    SIG = -np.log10(0.05)  # 1.301
    M = np.full((len(tids), 5), np.nan)
    for i, tid in enumerate(tids):
        for j, t in enumerate(TESTS):
            q = float(qmat[tid][TEST_COL[t]])
            val = -np.log10(max(q, 1e-6))
            if val >= SIG:
                M[i, j] = val
    ax = fig.add_subplot(gs[0, 1])
    cmap = LinearSegmentedColormap.from_list(
        "sig", ["#bfe3da", CB["green"], CB["blue"], "#08306b"])
    cmap.set_bad("#f0f0f0")  # non-significant cells
    im = ax.imshow(np.ma.masked_invalid(M.T), aspect="auto", cmap=cmap,
                   vmin=SIG, vmax=6.0, interpolation="nearest")
    for j in range(1, 5):  # thin separators between test rows
        ax.axhline(j - 0.5, color="white", lw=0.7)
    ax.set_yticks(range(5))
    ax.set_yticklabels([TEST_DISPLAY[t] for t in TESTS])
    ax.set_xticks([])
    ax.set_xlabel("38 conserved triplets (ordered by signature cluster)")
    ax.set_title("Per-triplet characterization")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(r"$-\log_{10}q$", fontsize=6.5)
    cb.ax.tick_params(labelsize=6)
    panel_label(ax, "b")

    # (c) cross-test Jaccard overlap
    ax = fig.add_subplot(gs[1, 0])
    im = ax.imshow(jac, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(np.arange(-0.5, 5, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 5, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", length=0)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{jac[i,j]:.2f}", ha="center", va="center",
                    fontsize=5.5, color="white" if jac[i, j] > 0.55 else CB["dark"])
    ax.set_xticks(range(5)); ax.set_yticks(range(5))
    ax.set_xticklabels([TEST_DISPLAY[t] for t in TESTS], rotation=40, ha="right")
    ax.set_yticklabels([TEST_DISPLAY[t] for t in TESTS])
    ax.set_title("Signature co-occurrence (Jaccard)")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.ax.tick_params(labelsize=6)
    panel_label(ax, "c")

    # (d) one mature-versus-early checkpoint-candidate comparison. This is a
    #     sensitivity result for a single pair, not a validated diagnostic
    #     across training seeds, model families or checkpoint trajectories.
    ax = fig.add_subplot(gs[1, 1])
    _ygrid(ax)
    names = ["v2_reference", "early10k"]
    labels = ["mature candidate\n(v2)", "early candidate\n(10k step)"]
    trips = [rows[n]["n_universal_triplets"] for n in names]
    cka = [rows[n]["mean_layer_cka"] for n in names]
    mr = [rows[n]["mean_abs_match_corr"] for n in names]
    xx = np.arange(2)
    bars = ax.bar(xx, trips, 0.46, color=[CB["green"], CB["orange"]],
                  zorder=3, edgecolor="white", linewidth=0.4)
    for b, v in zip(bars, trips):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.8, str(v),
                ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    ax.set_xticks(xx); ax.set_xticklabels(labels)
    ax.set_ylabel(r"triplets at $|r|\geq0.90$")
    ax.set_ylim(0, 44)
    ax.set_xlim(-0.6, 1.6)
    ax2 = ax.twinx()
    ax2.spines["top"].set_visible(False)
    ax2.plot(xx, cka, "o-", color=CB["blue"], lw=1.3, ms=4.5, zorder=4,
             label="layer CKA")
    ax2.plot(xx, mr, "s--", color=CB["purple"], lw=1.3, ms=4.5, zorder=4,
             label="match $|r|$")
    ax2.set_ylabel("CKA / match $|r|$")
    ax2.set_ylim(0, 1)
    ax2.legend(frameon=False, loc="upper right", fontsize=6,
               bbox_to_anchor=(1.0, 1.0))
    ax.text(0.5, 0.04, "one checkpoint pair; no run replicates",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=5.6,
            color="#6a4700",
            bbox={"boxstyle": "round,pad=0.22", "facecolor": "#fff3df",
                  "edgecolor": CB["orange"], "linewidth": 0.6})
    ax.set_title("Single checkpoint-pair sensitivity")
    panel_label(ax, "d")

    save(fig, "figure_02_sparse_readout_atlas")


# --------------------------------------------------------------------------
# Figure 3: N-terminal readouts with high unnormalized received attention
# --------------------------------------------------------------------------

AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"


def _k3_frequency_matrix(rows: list[dict]) -> np.ndarray:
    """Return amino-acid-by-k3-position frequencies from every saved row."""
    mers = [r["k3"].strip().upper() for r in rows]
    invalid = [m for m in mers if len(m) != 3 or any(a not in AA_ORDER for a in m)]
    if invalid:
        raise ValueError(f"invalid k3 values in saved top rows: {invalid[:5]}")
    matrix = np.zeros((len(AA_ORDER), 3), dtype=float)
    aa_index = {aa: i for i, aa in enumerate(AA_ORDER)}
    for mer in mers:
        for pos, aa in enumerate(mer):
            matrix[aa_index[aa], pos] += 1
    matrix /= len(mers)
    return matrix


def _draw_k3_frequency_summary(ax, rows: list[dict], title: str):
    matrix = _k3_frequency_matrix(rows)
    im = ax.imshow(matrix, aspect="auto", cmap="Blues", vmin=0, vmax=1,
                   interpolation="nearest")
    for i in range(matrix.shape[0]):
        for j in range(3):
            if matrix[i, j] >= 0.08:
                ax.text(j, i, f"{matrix[i, j]:.0%}", ha="center", va="center",
                        fontsize=4.7,
                        color="white" if matrix[i, j] >= 0.45 else CB["dark"])
    ax.set_xticks(range(3))
    ax.set_xticklabels(["k3 pos. 1", "k3 pos. 2", "k3 pos. 3"], fontsize=5.5)
    ax.set_yticks(range(len(AA_ORDER)))
    ax.set_yticklabels(list(AA_ORDER), fontsize=4.8)
    ax.set_xlabel("position within saved 3-mer", fontsize=6.2)
    ax.set_title(title, fontsize=7)
    return im


def figure3() -> None:
    sink = {r["triplet_id"]: r for r in read_tsv(CA / "attention_sink_subset_20260516/attention_sink_subset.tsv")}
    top_pos = read_tsv(CA / "triplet_characterization_20260515_nperm2000/top_firing_positions.tsv")
    targets = ["T011", "T018", "T023", "T025"]

    fig = plt.figure(figsize=(7.2, 5.6))
    gs = fig.add_gridspec(2, 3, hspace=0.5, wspace=0.45, height_ratios=[1, 0.95])

    # (a) Top-firing position distribution. The previous fraction bar chart was
    #     nearly binary (1/1/1 for T011/T018/T023, 0/0/0 for T025); plotting the
    #     saved positions directly makes the edge localization visible without
    #     implying a continuous effect size.
    ax = fig.add_subplot(gs[0, 0])
    colors = [CB["blue"], CB["green"], CB["orange"], CB["grey"]]
    by_triplet = {tid: [] for tid in targets}
    for row in top_pos:
        tid = row["triplet_id"]
        if tid in by_triplet:
            by_triplet[tid].append(row)
    for tid, rows in by_triplet.items():
        if len(rows) != 100:
            raise ValueError(f"expected 100 saved top rows for {tid}, found {len(rows)}")
    ax.axvspan(0, 0.20, color=CB["green"], alpha=0.11, lw=0, zorder=0)
    ax.axvline(0.20, color=CB["green"], lw=0.7, ls=":", zorder=1)
    for k, tid in enumerate(targets):
        rows = sorted(by_triplet[tid], key=lambda r: int(r["top_rank"]))
        xs = np.array([float(r["position_norm"]) for r in rows])
        y0 = len(targets) - 1 - k
        jitter = ((np.arange(len(xs)) % 17) - 8) / 8 * 0.085
        ax.scatter(xs, y0 + jitter, s=8, color=colors[k], alpha=0.58,
                   edgecolors="none", zorder=3)
        ax.plot([np.median(xs)], [y0], marker="D", ms=4.2, color=colors[k],
                mec="white", mew=0.45, zorder=4)
    ax.text(0.10, 3.58, "N-term\n20%", ha="center", va="top", fontsize=5.4,
            color=CB["green"])
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.55, 3.55)
    ax.set_yticks(range(len(targets)))
    ax.set_yticklabels(targets[::-1])
    ax.set_xlabel("normalized top-firing position")
    ax.set_ylabel("triplet")
    ax.set_title("Top-firing positions")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    panel_label(ax, "a")

    # (b) attention-received correlation
    ax = fig.add_subplot(gs[0, 1])
    _ygrid(ax)
    r = [float(sink[t]["attention_r"]) for t in targets]
    bars = ax.bar(targets, r, color=[CB["blue"], CB["green"], CB["orange"], CB["grey"]],
                  zorder=3, edgecolor="white", linewidth=0.3)
    for b, v in zip(bars, r):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}",
                ha="center", va="bottom", fontsize=6.5)
    ax.set_ylabel("attention-received corr. $r$")
    ax.set_ylim(0, 1.0)
    ax.set_title("Unnormalized received-attention\ncorrelation")
    panel_label(ax, "b", dx=-0.11, dy=1.10)

    # (c) descriptive N-terminal fractions among the saved top 100 rows. This
    #     panel intentionally avoids residue-event Fisher tests: the saved rows
    #     are ranked events rather than an independent residue-level sample.
    ax = fig.add_subplot(gs[0, 2])
    _ygrid(ax)
    fields = ["first2_fraction", "first5_fraction", "nterm20_fraction"]
    tlabels = ["first 2", "first 5", "N-term 20%"]
    x = np.arange(len(fields))
    offsets = np.linspace(-0.18, 0.18, len(targets))
    for k, tid in enumerate(targets):
        vals = [float(sink[tid][field]) for field in fields]
        ax.plot(x + offsets[k], vals, "o", color=colors[k], ms=4.6, label=tid,
                zorder=3, markeredgecolor="white", markeredgewidth=0.4)
    ax.set_xticks(x); ax.set_xticklabels(tlabels, rotation=15)
    ax.set_ylabel("fraction of saved top rows")
    ax.set_ylim(-0.04, 1.08)
    ax.set_title("N-terminal fractions\n(descriptive; $n=100$ each)")
    ax.legend(frameon=False, fontsize=5.8, loc="center right")
    panel_label(ax, "c", dx=-0.11, dy=1.10)

    # (d-f) amino-acid frequencies from all 100 saved k3 rows per triplet.
    # These are descriptive frequency summaries, not selected sequence logos.
    bottom_axes = []
    for k, tid in enumerate(["T011", "T018", "T023"]):
        ax = fig.add_subplot(gs[1, k])
        bottom_axes.append(ax)
        im = _draw_k3_frequency_summary(
            ax, by_triplet[tid], f"{tid} k3 frequency summary\n(all 100 saved rows)")
        if k == 0:
            ax.set_ylabel("amino acid", fontsize=6.2)
        panel_label(ax, ["d", "e", "f"][k], dx=-0.11, dy=1.16)

    cb = fig.colorbar(im, ax=bottom_axes, fraction=0.018, pad=0.025)
    cb.set_label("row fraction", fontsize=6.2)
    cb.ax.tick_params(labelsize=5.5)

    save(fig, "figure_03_nterminal_readouts")


# --------------------------------------------------------------------------
# Figure 4: calibrated negatives (steering + causal gates)
# --------------------------------------------------------------------------

def figure4() -> None:
    steer = json.loads((SB / "zymctrl_v2_onmanifold_direct_20260503.json").read_text())
    pc = steer["per_class"]
    order = ["lysozyme", "trypsin", "ADH", "catalase", "DNA_polymerase",
             "lipase", "kinase", "carbonic_anh"]
    disp = {"lysozyme": "lysozyme", "trypsin": "trypsin", "ADH": "ADH",
            "catalase": "catalase", "DNA_polymerase": "DNA pol.",
            "lipase": "lipase", "kinase": "kinase", "carbonic_anh": "carbonic anh."}

    fig = plt.figure(figsize=(7.2, 3.7))
    gs = fig.add_gridspec(1, 2, wspace=0.45, width_ratios=[1.05, 1.0])

    # (a) steering forest plot
    ax = fig.add_subplot(gs[0, 0])
    ax.grid(axis="x", zorder=0)
    ax.set_axisbelow(True)
    y = np.arange(len(order))[::-1]
    for yi, cls in zip(y, order):
        st = pc[cls]["statistics"]
        diff = st["obs_diff"]
        lo, hi = st["ci95"]
        p = st["perm_p"]
        col = CB["red"] if (p < 0.05 and diff > 0) else CB["grey"]
        ax.errorbar(diff, yi, xerr=[[diff - lo], [hi - diff]], fmt="o", color=col,
                    ms=4.5, elinewidth=1.2, capsize=2.0, capthick=1.0,
                    mec="white", mew=0.4, zorder=3)
        ax.text(hi + 0.012, yi, f"$P$={p:.2f}", va="center", fontsize=5.6,
                color=CB["dark"])
    ax.axvline(0, color=CB["dark"], lw=0.7, ls="--", zorder=2)
    ax.set_yticks(y); ax.set_yticklabels([disp[c] for c in order])
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_xlabel("steered − unsteered motif/composition\nheuristic (95% CI)")
    ax.set_xlim(-0.16, 0.26)
    ax.set_title("Heuristic steering-score effect")
    panel_label(ax, "a")

    # (b) All available target rows and matched random-control rows for the four
    #     intervention families. The previous panel plotted one selected maximum
    #     per family; this version retains every reported mean and 95% CI.
    ax = fig.add_subplot(gs[0, 1])
    ax.grid(axis="y", which="major", zorder=0)
    ax.set_axisbelow(True)

    effect_specs = [
        ("feature patch", CA / "attention_sink_causal_ablation_20260517/condition_summary.tsv",
         {"target"}, {"random"}),
        ("single head", CA / "attention_head_sink_ablation_20260518/condition_summary.tsv",
         {"target"}, {"random_head"}),
        ("top-8 set", CA / "attention_sink_set_ablation_20260518/condition_summary.tsv",
         {"sink_set"}, {"random_set"}),
        ("top-32 set", CA / "attention_sink_set_ablation_top32_20260518/condition_summary.tsv",
         {"sink_set"}, {"random_set"}),
    ]
    model_colors = {
        "protgpt2": CB["blue"],
        "zymctrl": CB["orange"],
        "progen2-medium": CB["green"],
    }
    model_order = {name: i for i, name in enumerate(model_colors)}
    group_counts = []

    for xi, (_, path, target_kinds, random_kinds) in enumerate(effect_specs):
        rows = read_tsv(path)
        targets = [r for r in rows if r["condition_kind"] in target_kinds]
        randoms = [r for r in rows if r["condition_kind"] in random_kinds]
        targets.sort(key=lambda r: (model_order[r["model"]], r["condition"]))
        randoms.sort(key=lambda r: (model_order[r["model"]], r["condition"]))
        group_counts.append((len(targets), len(randoms)))

        for kind, selected, offsets in (
            ("target", targets, np.linspace(-0.28, -0.04, len(targets))),
            ("random", randoms, np.linspace(0.04, 0.28, len(randoms))),
        ):
            for row, offset in zip(selected, offsets):
                mean = float(row["mean_delta_nll_pos2_10"])
                low = float(row["delta_nll_pos2_10_ci_low"])
                high = float(row["delta_nll_pos2_10_ci_high"])
                if not np.all(np.isfinite([mean, low, high])):
                    raise ValueError(f"non-finite intervention effect in {path}: {row}")
                col = model_colors[row["model"]]
                ax.errorbar(
                    xi + offset, mean,
                    yerr=[[mean - low], [high - mean]],
                    fmt="o" if kind == "target" else "x",
                    color=col, ms=3.5 if kind == "target" else 3.2,
                    markeredgewidth=0.65, elinewidth=0.6,
                    capsize=1.2, alpha=0.9 if kind == "target" else 0.48,
                    zorder=4 if kind == "target" else 3,
                )

    ax.axhline(0, color=CB["dark"], lw=0.75, zorder=2)
    ax.axhline(0.5, color=CB["red"], lw=0.75, ls=":", zorder=2)
    ax.text(3.48, 0.51, "0.5-nat target gate", ha="right", va="bottom",
            fontsize=5.3, color=CB["red"])
    ax.set_yscale("symlog", linthresh=0.002, linscale=0.8)
    ax.set_ylim(-0.48, 0.65)
    ax.set_yticks([-0.3, -0.03, -0.003, 0, 0.003, 0.03, 0.3])
    ax.set_yticklabels(["−0.3", "−0.03", "−0.003", "0", "0.003", "0.03", "0.3"])
    x = np.arange(len(effect_specs))
    labels = [
        f"{label}\n{nt} T / {nr} R"
        for (label, *_), (nt, nr) in zip(effect_specs, group_counts)
    ]
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=5.4)
    ax.set_ylabel(r"$\Delta$NLL, positions 2–10 (nats)")
    from matplotlib.lines import Line2D
    model_handles = [
        Line2D([0], [0], marker="o", ls="none", color=col, label=MLAB[name],
               markersize=3.8)
        for name, col in model_colors.items()
    ]
    kind_handles = [
        Line2D([0], [0], marker="o", ls="none", color=CB["dark"],
               label="target (T)", markersize=3.8),
        Line2D([0], [0], marker="x", ls="none", color=CB["dark"],
               label="matched random (R)", markersize=3.8),
    ]
    ax.legend(handles=model_handles + kind_handles, frameon=False, fontsize=5.2,
              loc="upper left", ncol=2, columnspacing=0.6, handletextpad=0.25)
    ax.set_title("All target and matched-random rows\n"
                 "($n=200$ sequences per row; mean and 95% CI)")
    panel_label(ax, "b")

    save(fig, "figure_04_negative_interventions")


# --------------------------------------------------------------------------
# Figure 5: generated lysozyme structural validation
# --------------------------------------------------------------------------

DRUG = REPO / "r2_interpretability_transfer/results/drug_design"
ECM = REPO / "r2_interpretability_transfer/results/ec_metrics"
RENDER_DIR = DRUG / "pymol_renders"
DISPLAY_ESMFOLD = DRUG / "ec_lysozyme_esmfold_metrics.json"
STEERED_ESMFOLD = DRUG / "ec_lysozyme_esmfold_metrics_v2_20260425_r2_v2_1gpu.json"
UNSTEERED_ESMFOLD = DRUG / "ec_lysozyme_unsteered_esmfold_metrics_v2_20260425_r2_v2_1gpu.json"
LEAD_GENERATION = DRUG / "ec_lysozyme_leads_v2.json"
FOLDSEEK_GENERATED = ECM / "foldseek_generated_lysozyme_20260507.json"
# AlphaFold/ESMFold pLDDT confidence palette (discrete bins).
PLDDT_BINS = [("#FF7D45", "<50"), ("#FFDB13", "50–70"),
              ("#65CBF3", "70–90"), ("#0053D6", ">90")]
# Post-hoc display subset rendered as pLDDT cartoons by PyMOL. This subset is
# explicitly labelled in the figure and is not presented as representative.
STRUCT_PANELS = ["lead_0004", "lead_0009", "lead_0008"]


def _load_crop(png_path: Path):
    """Load a PyMOL render and crop transparent/white margins."""
    img = plt.imread(png_path)
    if img.ndim == 3 and img.shape[2] == 4:
        mask = img[..., 3] > 0.04
    else:
        mask = (img[..., :3] < 0.985).any(-1)
    ys, xs = np.where(mask)
    if ys.size == 0:
        return img
    pad = 6
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, img.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, img.shape[1])
    return img[y0:y1, x0:x1]


def figure5() -> None:
    from matplotlib.patches import Patch

    fig = plt.figure(figsize=(7.2, 2.95))
    gs = fig.add_gridspec(
        2, 4,
        wspace=0.30,
        hspace=0.08,
        height_ratios=[1.0, 0.16],
        width_ratios=[1, 1, 1, 1.05],
    )

    display_data = json.loads(DISPLAY_ESMFOLD.read_text())
    display_rows = {
        Path(row["pdb_path"]).stem: row for row in display_data["per_sequence"]
    }
    for name in STRUCT_PANELS:
        if name not in display_rows:
            raise ValueError(f"rendered structure {name} absent from {DISPLAY_ESMFOLD}")

    for k, name in enumerate(STRUCT_PANELS):
        row = display_rows[name]
        ax = fig.add_subplot(gs[0, k])
        png = RENDER_DIR / f"{name}.png"
        ax.imshow(_load_crop(png), interpolation="lanczos")
        ax.set_axis_off()
        ax.set_title(
            f"selected lead {int(name.split('_')[-1])}\n"
            f"{int(row['seq_len'])} aa $\\cdot$ pLDDT {row['mean_plddt']:.0f}",
            fontsize=7, pad=2,
        )
        panel_label(ax, ["a", "b", "c"][k], dx=0.04, dy=1.02)

    # Discrete pLDDT legend aligned to the same three-column region as panels a--c.
    handles = [Patch(facecolor=c, edgecolor="none", label=l) for c, l in PLDDT_BINS]
    legend_ax = fig.add_subplot(gs[1, 0:3])
    legend_ax.set_axis_off()
    legend_ax.legend(handles=handles, title="pLDDT", ncol=4, frameon=False,
                     fontsize=6, title_fontsize=6.5, handlelength=1.0,
                     columnspacing=1.0, loc="upper center", borderaxespad=0.0)
    legend_ax.text(
        0.5, 0.02,
        "post-hoc display subset: 3 of 10 rank-selected steered leads",
        transform=legend_ax.transAxes, ha="center", va="bottom", fontsize=5.5,
        color="#6a4700",
    )

    # (d) steered vs unsteered structural quality (v2 aggregate)
    ax = fig.add_subplot(gs[:, 3])
    _ygrid(ax)
    se = json.loads(STEERED_ESMFOLD.read_text())["aggregate"]
    ue = json.loads(UNSTEERED_ESMFOLD.read_text())["aggregate"]
    generation = json.loads(LEAD_GENERATION.read_text())
    foldseek = json.loads(FOLDSEEK_GENERATED.read_text())["sets"]
    se_tm = foldseek["steered_leads"]["summary"]["mean_top_alntmscore"]
    ue_tm = foldseek["unsteered_baseline"]["summary"]["mean_top_alntmscore"]
    se_n = int(se["n_folded"])
    ue_n = int(ue["n_folded"])
    n_generated = int(generation["n_generated"])
    n_unsteered_generated = len(generation["unsteered_baseline"])
    metrics = ["mean pLDDT/100", "Foldseek TM", "confident frac."]
    steered = [se["mean_plddt_dist"]["mean"] / 100.0, se_tm, se["frac_globally_confident"]]
    unsteered = [ue["mean_plddt_dist"]["mean"] / 100.0, ue_tm, ue["frac_globally_confident"]]
    x = np.arange(3)
    sb = ax.bar(x - 0.2, steered, 0.38, color=CB["orange"],
                label=f"top-ranked steered ($n={se_n}/{n_generated}$)",
                zorder=3, edgecolor="white", linewidth=0.4)
    ub = ax.bar(x + 0.2, unsteered, 0.38, color=CB["blue"],
                label=f"evaluated unsteered ($n={ue_n}/{n_unsteered_generated}$)",
                zorder=3, edgecolor="white", linewidth=0.4)
    for bars in (sb, ub):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015,
                    f"{b.get_height():.2f}", ha="center", va="bottom", fontsize=5.0)
    ax.set_xticks(x); ax.set_xticklabels(metrics, rotation=18, ha="right", fontsize=6)
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.18)
    ax.legend(frameon=False, fontsize=5.2, loc="lower center",
              bbox_to_anchor=(0.5, 1.005), borderaxespad=0.0)
    ax.set_title("Structural scores", fontsize=7.5, y=1.20, pad=2)
    panel_label(ax, "d", dx=-0.22, dy=1.04)

    save(fig, "figure_05_sequence_structure_checks")


# --------------------------------------------------------------------------
# Figure 6: representation-recoverability audit
# --------------------------------------------------------------------------

PROBES_V2 = REPO / "r2_interpretability_transfer/evidence/recoverability_audit_20260605_1250/probes_v2/probe_results.json"
EXPANDED_PROBES = (
    REPO / "r2_interpretability_transfer/evidence/recoverability_audit_20260605_1250/"
    "expanded_dictionary_probe_summary_20260612.json"
)
MCOL = {"protgpt2": CB["blue"], "zymctrl": CB["orange"], "progen2-medium": CB["green"]}
MLAB = {"protgpt2": "ProtGPT2", "zymctrl": "ZymCTRL", "progen2-medium": "ProGen2-med"}


def figure6() -> None:
    from matplotlib.lines import Line2D
    if not PROBES_V2.exists():
        print(f"skip figure6: {PROBES_V2} not found")
        return
    data = json.loads(PROBES_V2.read_text())["models"]
    expanded_data = json.loads(EXPANDED_PROBES.read_text())
    if expanded_data.get("status") != "exploratory":
        raise ValueError(f"expanded-dictionary run must be labelled exploratory: {EXPANDED_PROBES}")
    expanded_rows = {
        (row["model"], row["task"]): row
        for row in expanded_data["rows"] if row.get("figure6_panel_c")
    }

    def val(model, task, key):
        r = data.get(model, {}).get(task)
        if not r or r.get("status") != "ok":
            return None
        if key == "C":
            return r["ceiling"]["skill"]
        if key == "F":
            return r["floor_same_layer"]["skill"]
        if key == "rho":
            return r["recovery_ratio"]
        return None

    fig = plt.figure(figsize=(7.2, 2.9))
    gs = fig.add_gridspec(1, 3, wspace=0.46, width_ratios=[1.05, 0.85, 1.2])

    # (a) Same linear probes on each architecture's CLT-input tensor and code.
    #     This is a task-specific comparison, not whole-representation fidelity.
    ax = fig.add_subplot(gs[0, 0])
    ax.plot([0, 1], [0, 1], color=CB["dark"], lw=0.8, zorder=1)
    tasks = [("ec_topclass", "o", "EC"), ("pfam_family", "s", "Pfam"),
             ("residue_ss", "^", "res-SS"), ("decoder_ec", "D", "dec-EC")]
    for m, col in MCOL.items():
        for task, mk, _ in tasks:
            C, F = val(m, task, "C"), val(m, task, "F")
            if C is None or F is None:
                continue
            ax.scatter(C, F, s=26, marker=mk, color=col, edgecolor="white",
                       linewidth=0.4, zorder=3)
    ax.set_xlim(0, 1.0); ax.set_ylim(0, 1.0)
    ax.set_xlabel("skill $C$ (architecture-specific CLT input)")
    ax.set_ylabel("floor skill $F$ (sparse codes)")
    ax.set_title("CLT-input vs sparse-code probes")
    ax.text(0.96, 0.99, "$y=x$", fontsize=6, ha="right", va="top",
            color=CB["dark"], transform=ax.transAxes)
    mh = [Line2D([0], [0], marker="o", ls="none", color=c, label=MLAB[m], mec="white", mew=0.4)
          for m, c in MCOL.items()]
    th = [Line2D([0], [0], marker=mk, ls="none", color=CB["dark"], label=lab) for _, mk, lab in tasks]
    leg1 = ax.legend(handles=mh, frameon=False, fontsize=5.4, loc="upper left",
                     handletextpad=0.2, labelspacing=0.2)
    ax.add_artist(leg1)
    ax.legend(handles=th, frameon=False, fontsize=5.4, loc="lower right",
              handletextpad=0.2, labelspacing=0.2, ncol=2, columnspacing=0.8)
    panel_label(ax, "a", dx=-0.14, dy=1.06)

    # (b) EC family confound: family-disjoint vs stratified ceiling
    ax = fig.add_subplot(gs[0, 1]); _ygrid(ax)
    models = ["protgpt2", "zymctrl", "progen2-medium"]
    x = np.arange(len(models)); w = 0.38
    fd = [val(m, "ec_topclass", "C") for m in models]
    st = [val(m, "ec_topclass_stratified", "C") for m in models]
    ax.bar(x - w / 2, fd, w, color=CB["sky"], label="family-disjoint",
           zorder=3, edgecolor="white", linewidth=0.4)
    ax.bar(x + w / 2, st, w, color=CB["purple"], label="stratified",
           zorder=3, edgecolor="white", linewidth=0.4)
    ax.set_xticks(x); ax.set_xticklabels([MLAB[m] for m in models], rotation=18, ha="right")
    ax.set_ylabel("EC top-class skill $C$")
    ax.set_ylim(0, 0.7)
    ax.legend(frameon=False, fontsize=6, loc="upper left")
    ax.set_title("EC probe skill depends\non cross-validation split")
    panel_label(ax, "b")

    # (c) Selected reference-versus-wider comparisons from the exploratory
    #     expanded-dictionary run. Values are loaded from a machine-readable
    #     evidence file rather than embedded in this plotting script.
    ax = fig.add_subplot(gs[0, 2]); _ygrid(ax)
    task_labels = {
        "ec_topclass": "EC",
        "pfam_family": "Pfam",
        "decoder_ec": "dec-EC",
    }
    short_model = {
        "protgpt2": "PGPT2", "zymctrl": "Zym", "progen2-medium": "PGen2",
    }
    pairs = [
        (model, task, f"{short_model[model]}\n{task_labels[task]}")
        for model, task in expanded_rows
    ]
    x = np.arange(len(pairs)); w = 0.38
    orig = [val(m, t, "rho") for m, t, _ in pairs]
    exp = [expanded_rows[(m, t)]["recovery_ratio"] for m, t, _ in pairs]
    ax.bar(x - w / 2, orig, w, color=CB["grey"], label="reference dict.",
           zorder=3, edgecolor="white", linewidth=0.4)
    ax.bar(x + w / 2, exp, w, color=CB["blue"], label="wider dict. (exploratory)",
           zorder=3, edgecolor="white", linewidth=0.4)
    ax.set_xticks(x); ax.set_xticklabels([lab for *_, lab in pairs], fontsize=5.4)
    ax.set_ylabel(r"recovery ratio $\rho=F/C$")
    ax.set_ylim(0, 1.1)
    ax.legend(frameon=False, fontsize=5.5, loc="lower left")
    ax.set_title("Exploratory wider-dictionary run\n(selected task–model pairs)")
    panel_label(ax, "c")

    save(fig, "figure_06_recoverability_audit")


def main() -> None:
    nature_style()
    figure1()
    figure2()
    figure3()
    figure4()
    figure5()
    figure6()
    print("All figures written to", OUT)


if __name__ == "__main__":
    main()
