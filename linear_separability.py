# linear_separability.py
#
# Research question answered here:
#   Are ArmoRM's reward head scores linearly decodable from the model's
#   internal hidden states?
#
#   We train one logistic regression probe per head. The probe takes the
#   final-layer hidden state of a response as input and tries to predict
#   whether that response scored above or below median on that head.
#
#   High probe accuracy = the concept is linearly encoded in the representations.
#   Low probe accuracy  = the head score is not reflected in the geometry of
#                         the hidden space — the label may be superficial.
#
# This is the mechanistic interpretability core of the project.
# It connects directly to the probing literature (Alain & Bengio 2016,
# Belinkov 2022) that Rachel will recognize immediately.
#
# Outputs:
#   outputs/linear_separability_results.json  — accuracy per head
#   outputs/linear_separability_bar.png       — bar chart of probe accuracies
#
# Usage:
#   python linear_separability.py
#   (Requires outputs/ populated by extract_scores_and_embeddings.py)

import numpy as np
import json
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from utils import load_outputs, HEAD_NAMES, SAVE_DIR

RESULTS_PATH = os.path.join(SAVE_DIR, "linear_separability_results.json")
BAR_PATH     = os.path.join(SAVE_DIR, "linear_separability_bar.png")

N_FOLDS    = 5      # 5-fold cross-validation — robust accuracy estimate
MAX_ITER   = 1000   # Logistic regression convergence iterations
N_COMPONENTS = 50  # PCA dimensionality reduction before probing (memory + speed)


# ── Block 1: Dimensionality reduction ─────────────────────────────────────────

def reduce_dimensions(chosen_hidden, rejected_hidden, n_components=N_COMPONENTS):
    """
    Hidden states from Llama3-8B are 4096-dimensional.
    Running logistic regression directly on 4096 dims with 1600 samples
    is slow and prone to overfitting.

    We apply PCA to reduce to 50 dimensions, retaining the directions of
    maximum variance. This is standard practice in probing studies.

    Fits PCA on the full pool (chosen + rejected), transforms both.
    Returns reduced arrays and the fitted PCA object.
    """
    from sklearn.decomposition import PCA

    all_hidden = np.vstack([chosen_hidden, rejected_hidden])  # (2N, 4096)

    print(f"  Fitting PCA: {all_hidden.shape[1]}d -> {n_components}d ...")
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(all_hidden)

    variance_explained = pca.explained_variance_ratio_.sum()
    print(f"  Variance explained by {n_components} components: {variance_explained:.1%}")

    chosen_reduced   = pca.transform(chosen_hidden)    # (N, 50)
    rejected_reduced = pca.transform(rejected_hidden)  # (N, 50)

    return chosen_reduced, rejected_reduced, pca


# ── Block 2: Build binary labels per head ─────────────────────────────────────

def build_probe_labels(chosen_scores, rejected_scores):
    """
    For each head, we need a binary label: did this response score
    above or below the median on this head?

    We pool chosen and rejected scores, compute the median per head,
    then label each response 1 (above median) or 0 (below median).

    Returns:
        all_labels : (2N, num_heads) array of 0/1 labels
    """
    all_scores = np.vstack([chosen_scores, rejected_scores])  # (2N, num_heads)
    medians    = np.median(all_scores, axis=0)                 # (num_heads,)

    # Binary: 1 if above median, 0 if below
    all_labels = (all_scores > medians).astype(int)            # (2N, num_heads)

    return all_labels, medians


# ── Block 3: Train and evaluate probes ────────────────────────────────────────

def probe_all_heads(chosen_reduced, rejected_reduced, all_labels):
    """
    For each of the 8 reward heads, trains a logistic regression probe
    using 5-fold cross-validation.

    The probe input  : 50-dim PCA-reduced hidden state
    The probe target : binary label (above/below median score on this head)

    Cross-validation gives us 5 accuracy estimates per head.
    We report mean and std — std tells us how stable the probe is.

    Baseline accuracy is 50% (random binary classification).
    A well-encoded concept should probe at 65%+ comfortably.
    """
    all_hidden = np.vstack([chosen_reduced, rejected_reduced])  # (2N, 50)
    num_heads  = all_labels.shape[1]

    results = []
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    for head_idx, head_name in enumerate(HEAD_NAMES):
        labels = all_labels[:, head_idx]   # (2N,) binary labels for this head
        fold_accuracies = []

        for fold, (train_idx, test_idx) in enumerate(skf.split(all_hidden, labels)):
            X_train, X_test = all_hidden[train_idx], all_hidden[test_idx]
            y_train, y_test = labels[train_idx],     labels[test_idx]

            # Standardize features — logistic regression is sensitive to scale
            scaler  = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test  = scaler.transform(X_test)

            clf = LogisticRegression(max_iter=MAX_ITER, random_state=42, C=1.0)
            clf.fit(X_train, y_train)

            acc = accuracy_score(y_test, clf.predict(X_test))
            fold_accuracies.append(acc)

        mean_acc = float(np.mean(fold_accuracies))
        std_acc  = float(np.std(fold_accuracies))

        results.append({
            "head":          head_name,
            "mean_accuracy": round(mean_acc, 4),
            "std_accuracy":  round(std_acc, 4),
            "fold_accuracies": [round(a, 4) for a in fold_accuracies],
            "interpretation": interpret_accuracy(mean_acc)
        })

        print(f"  {head_name:<22}  acc = {mean_acc:.3f} ± {std_acc:.3f}  |  {interpret_accuracy(mean_acc)}")

    return results


# ── Block 4: Interpret accuracy ───────────────────────────────────────────────

def interpret_accuracy(acc):
    """
    Plain-language interpretation of probe accuracy.
    Baseline is 50% (chance). These thresholds are conventional in probing lit.
    """
    if acc >= 0.75:
        return "strongly linearly encoded"
    elif acc >= 0.65:
        return "moderately linearly encoded"
    elif acc >= 0.55:
        return "weakly encoded — noisy signal"
    else:
        return "not linearly encoded — near chance"


# ── Block 5: Plot bar chart ───────────────────────────────────────────────────

def plot_bar(results):
    """
    Bar chart: one bar per head, height = mean probe accuracy.
    Error bars = ±1 std across folds.
    Red dashed line at 0.5 = chance baseline.

    Bars above 0.65 are the positive finding: those heads are linearly
    encoded in the hidden states. Bars near 0.5 are the negative finding:
    those heads are not mechanistically grounded in the representations.
    """
    names  = [r["head"] for r in results]
    accs   = [r["mean_accuracy"] for r in results]
    stds   = [r["std_accuracy"] for r in results]

    # Color by interpretation
    colors = []
    for acc in accs:
        if acc >= 0.75:
            colors.append("#2ecc71")   # green — strongly encoded
        elif acc >= 0.65:
            colors.append("#f39c12")   # orange — moderately encoded
        else:
            colors.append("#e74c3c")   # red — weak or not encoded

    fig, ax = plt.subplots(figsize=(11, 5))

    bars = ax.bar(names, accs, yerr=stds, capsize=4,
                  color=colors, edgecolor="black", linewidth=0.7, alpha=0.85)

    # Chance baseline
    ax.axhline(0.5, color="red", linestyle="--", linewidth=1.2, label="Chance baseline (0.50)")

    # 0.65 threshold line
    ax.axhline(0.65, color="gray", linestyle=":", linewidth=1.0, label="Moderate encoding threshold (0.65)")

    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("Probe Accuracy (5-fold CV)", fontsize=11)
    ax.set_title(
        "Linear Separability of ArmoRM Reward Heads\n"
        "Logistic Regression Probes on Final-Layer Hidden States (PCA-50)",
        fontsize=12, pad=15
    )
    ax.legend(fontsize=9)
    plt.xticks(rotation=30, ha="right", fontsize=9)
    plt.tight_layout()
    plt.savefig(BAR_PATH, dpi=150)
    plt.close()

    print(f"Bar chart saved -> {BAR_PATH}")


# ── Block 6: Save results ─────────────────────────────────────────────────────

def save_results(results, medians, variance_explained):
    """
    Saves probe results to JSON.
    Includes per-head accuracies, medians used for binarization,
    PCA variance explained, and a plain-language summary.
    """
    strongly_encoded   = [r["head"] for r in results if r["mean_accuracy"] >= 0.75]
    moderately_encoded = [r["head"] for r in results if 0.65 <= r["mean_accuracy"] < 0.75]
    weakly_encoded     = [r["head"] for r in results if r["mean_accuracy"] < 0.65]

    output = {
        "analysis":           "linear_separability",
        "method":             f"Logistic regression probes, {N_FOLDS}-fold CV, PCA-{N_COMPONENTS}",
        "pca_variance_explained": round(float(variance_explained), 4),
        "chance_baseline":    0.50,
        "head_medians":       {HEAD_NAMES[i]: round(float(medians[i]), 4) for i in range(len(HEAD_NAMES))},
        "probe_results":      results,
        "summary": {
            "strongly_encoded":   strongly_encoded,
            "moderately_encoded": moderately_encoded,
            "weakly_encoded":     weakly_encoded
        }
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Results saved -> {RESULTS_PATH}")
    return output


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading outputs ...")
    chosen_scores, rejected_scores, chosen_hidden, rejected_hidden, chosen_gating, rejected_gating, meta = load_outputs()

    print("\nReducing hidden state dimensions with PCA ...")
    chosen_reduced, rejected_reduced, pca = reduce_dimensions(chosen_hidden, rejected_hidden)
    variance_explained = pca.explained_variance_ratio_.sum()

    print("\nBuilding binary labels (above/below median per head) ...")
    all_labels, medians = build_probe_labels(chosen_scores, rejected_scores)

    print(f"\nTraining logistic regression probes ({N_FOLDS}-fold CV) ...")
    print(f"{'Head':<22}  {'Accuracy':<20}  Interpretation")
    print("-" * 65)
    results = probe_all_heads(chosen_reduced, rejected_reduced, all_labels)

    print("\nPlotting bar chart ...")
    plot_bar(results)

    print("\nSaving results ...")
    output = save_results(results, medians, variance_explained)

    print("\n── Linear Separability Summary ───────────────────────────────────")
    print(f"Strongly encoded   (≥0.75) : {output['summary']['strongly_encoded']}")
    print(f"Moderately encoded (≥0.65) : {output['summary']['moderately_encoded']}")
    print(f"Weakly encoded     (<0.65) : {output['summary']['weakly_encoded']}")
    print("\nHow to read this for Rachel:")
    print("  A head that probes well is mechanistically present in the representations.")
    print("  A head that probes near chance is a label without a learned concept behind it.")
    print("  Weak probing + high reward weight = optimizing for a ghost.")


if __name__ == "__main__":
    main()
