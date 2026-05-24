# disentanglement_analysis.py
# Tests whether ArmoRM's reward heads are statistically independent.
# If "helpfulness" and "coherence" correlate strongly, they're the same signal with different labels —
# a disentanglement failure. Requires outputs/ from extract_scores_and_embeddings.py.

import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from utils import load_outputs, HEAD_NAMES, SAVE_DIR
import os

HEATMAP_PATH = os.path.join(SAVE_DIR, "disentanglement_heatmap.png")
REPORT_PATH  = os.path.join(SAVE_DIR, "disentanglement_report.json")

# Correlation threshold above which two heads are flagged as entangled
ENTANGLEMENT_THRESHOLD = 0.7


# ---

def compute_correlation_matrix(chosen_scores, rejected_scores):
    """
    Pools chosen + rejected (2N, num_heads) and computes pairwise Spearman rank correlation.
    Spearman rather than Pearson since reward scores aren't normally distributed.
    """
    # pool both splits — correlation should hold over the full distribution, not just chosen
    all_scores = np.vstack([chosen_scores, rejected_scores])  # (2N, num_heads)

    num_heads = all_scores.shape[1]
    corr_matrix = np.zeros((num_heads, num_heads))
    pval_matrix = np.zeros((num_heads, num_heads))

    for i in range(num_heads):
        for j in range(num_heads):
            corr, pval = spearmanr(all_scores[:, i], all_scores[:, j])
            corr_matrix[i, j] = corr
            pval_matrix[i, j] = pval

    return corr_matrix, pval_matrix


# ---

def find_entangled_pairs(corr_matrix, threshold=ENTANGLEMENT_THRESHOLD):
    """
    Returns upper-triangle pairs where |corr| > threshold — heads that activate together
    and are therefore not measuring distinct reward dimensions.
    """
    num_heads = corr_matrix.shape[0]
    entangled = []

    for i in range(num_heads):
        for j in range(i + 1, num_heads):
            corr = corr_matrix[i, j]
            if abs(corr) > threshold:
                entangled.append({
                    "head_a":      HEAD_NAMES[i],
                    "head_b":      HEAD_NAMES[j],
                    "correlation": round(float(corr), 4)
                })

    entangled.sort(key=lambda x: abs(x["correlation"]), reverse=True)
    return entangled


# ---

def plot_heatmap(corr_matrix, entangled_pairs):
    """Saves pairwise Spearman correlation heatmap. Off-diagonal red cells = entangled heads."""
    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        corr_matrix,
        xticklabels=HEAD_NAMES,
        yticklabels=HEAD_NAMES,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",       # red = positive correlation, blue = negative
        vmin=-1, vmax=1,
        linewidths=0.5,
        ax=ax
    )

    ax.set_title(
        "ArmoRM Reward Head Pairwise Correlations (Spearman)\n"
        "Cells near ±1.0 indicate entangled (non-independent) bases",
        fontsize=12, pad=15
    )

    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(HEATMAP_PATH, dpi=150)
    plt.close()

    print(f"Heatmap saved -> {HEATMAP_PATH}")


# ---

def save_report(corr_matrix, pval_matrix, entangled_pairs):
    """Saves full correlation matrix + flagged pairs to JSON."""
    # mean abs off-diagonal as a scalar disentanglement summary
    num_heads = corr_matrix.shape[0]
    mask = ~np.eye(num_heads, dtype=bool)
    mean_off_diag = float(np.abs(corr_matrix[mask]).mean())

    finding = (
        "Bases are well-disentangled." if mean_off_diag < 0.3
        else "Moderate entanglement detected — some bases are not independent."
        if mean_off_diag < 0.6
        else "High entanglement — reward bases are not meaningfully distinct."
    )

    report = {
        "analysis":                "disentanglement",
        "method":                  "Spearman rank correlation",
        "entanglement_threshold":  ENTANGLEMENT_THRESHOLD,
        "mean_abs_off_diag_corr":  round(mean_off_diag, 4),
        "finding":                 finding,
        "entangled_pairs":         entangled_pairs,
        "full_correlation_matrix": {
            HEAD_NAMES[i]: {
                HEAD_NAMES[j]: round(float(corr_matrix[i, j]), 4)
                for j in range(len(HEAD_NAMES))
            }
            for i in range(len(HEAD_NAMES))
        }
    }

    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Report saved  -> {REPORT_PATH}")
    return report


# ---

def main():
    print("Loading outputs ...")
    chosen_scores, rejected_scores, chosen_hidden, rejected_hidden, chosen_gating, rejected_gating, meta = load_outputs()

    print("\nComputing Spearman correlations between reward heads ...")
    corr_matrix, pval_matrix = compute_correlation_matrix(chosen_scores, rejected_scores)

    print("\nFlagging entangled pairs (|corr| > {}) ...".format(ENTANGLEMENT_THRESHOLD))
    entangled_pairs = find_entangled_pairs(corr_matrix)

    print("\nPlotting heatmap ...")
    plot_heatmap(corr_matrix, entangled_pairs)

    print("\nSaving report ...")
    report = save_report(corr_matrix, pval_matrix, entangled_pairs)

    print("\n── Disentanglement Findings ──────────────────────────────────────")
    print(f"Mean abs off-diagonal correlation : {report['mean_abs_off_diag_corr']}")
    print(f"Finding                           : {report['finding']}")

    if entangled_pairs:
        print(f"\nEntangled pairs (|corr| > {ENTANGLEMENT_THRESHOLD}):")
        for pair in entangled_pairs:
            print(f"  {pair['head_a']:<22} <-> {pair['head_b']:<22}  r = {pair['correlation']:+.4f}")
    else:
        print(f"\nNo pairs exceed threshold {ENTANGLEMENT_THRESHOLD}. Bases appear independent.")

    print("\nHow to read this for Rachel:")
    print("  High correlation between two heads = they activate together = not separate bases.")
    print("  If 'helpfulness' and 'coherence' both spike on the same responses,")
    print("  they are not measuring different things — the label decomposition is misleading.")


if __name__ == "__main__":
    main()
