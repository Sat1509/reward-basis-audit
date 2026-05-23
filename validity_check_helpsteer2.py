"""
validity_check_helpsteer2.py

Runs validity analysis on ArmoRM head scores computed on HelpSteer2 pairs.
Two analyses:

  ANALYSIS 1 — Overall (TYPE A pairs):
    Same test as validity_check.py on hh-rlhf.
    Preference accuracy, Cohen's d, Wilcoxon test per head.
    Direct comparison to hh-rlhf results.

  ANALYSIS 2 — Dimension-targeted (TYPE B pairs):
    For each of the 5 HelpSteer2 dimensions, run validity on only the
    pairs where that dimension was the controlled variable.
    Key question: on pairs where helpfulness is the explicit difference,
    does the helpfulness head agree?

This is the test that exposes whether head failures on hh-rlhf are:
  (a) a dataset noise problem — heads recover on clean targeted pairs, OR
  (b) a genuine model problem — heads fail even when the signal is clean

If (a): the model is fine, just sensitive to dataset quality.
If (b): the labeled decomposition is genuinely misleading.
"""

import json
import numpy as np
from scipy import stats
import os

# ── CONFIG ──────────────────────────────────────────────────────────────────

INPUT_DIR = "outputs_helpsteer2"

HEAD_NAMES = [
    "helpfulness", "correctness", "coherence", "complexity", "verbosity",
    "safety", "instruction_following", "honesty", "truthfulness", "harmlessness",
    "readability", "depth", "creativity", "detail", "positivity",
    "clarity", "engagement", "conciseness", "relevance"
]

HS2_DIMS = ["helpfulness", "correctness", "coherence", "complexity", "verbosity"]


# ── BLOCK 1: LOAD OUTPUTS ──────────────────────────────────────────────────
# What: Load the saved numpy arrays and the pair metadata.
# Why: We need both the scores (to compute gaps) and the metadata
#      (to know which pairs are type A vs type B, and which dimension).

print("Loading HelpSteer2 outputs ...")
head_scores = np.load(os.path.join(INPUT_DIR, "head_scores.npy"))    # (N, 2, 19)
with open(os.path.join(INPUT_DIR, "metadata.json")) as f:
    meta = json.load(f)

pairs = meta["pairs"]
N = len(pairs)
print(f"  {N} total pairs")

# Split indices by type
type_a_indices = [i for i, p in enumerate(pairs) if p["pair_type"] == "overall"]
type_b_indices = {
    dim: [i for i, p in enumerate(pairs) if p["pair_type"] == f"targeted_{dim}"]
    for dim in HS2_DIMS
}

print(f"  TYPE A (overall): {len(type_a_indices)} pairs")
for dim in HS2_DIMS:
    print(f"  TYPE B ({dim}): {len(type_b_indices[dim])} pairs")


# ── BLOCK 2: VALIDITY FUNCTION ──────────────────────────────────────────────
# What: Given a set of (chosen, rejected) score pairs for one head,
#       compute preference accuracy, mean gap, Cohen's d, and p-value.
# Why: Preference accuracy is the ground truth test — what fraction of pairs
#      did the head rank correctly? Cohen's d measures effect size.
#      Wilcoxon tests whether the gap distribution is consistently positive.
# Same implementation as validity_check.py — identical metrics, different data.

def compute_validity(chosen_scores, rejected_scores):
    """
    chosen_scores, rejected_scores: 1D arrays of shape (n_pairs,)
    Returns: acc, mean_gap, cohen_d, p_value, interpretation
    """
    gaps = chosen_scores - rejected_scores
    n = len(gaps)
    
    # Preference accuracy: fraction where chosen > rejected
    acc = (gaps > 0).mean()
    mean_gap = gaps.mean()
    
    # Cohen's d: effect size (mean gap / std of gaps)
    std = gaps.std()
    cohen_d = mean_gap / std if std > 0 else 0.0
    
    # Wilcoxon signed-rank test: is the gap distribution significantly positive?
    if n < 10 or (gaps == 0).all():
        p_value = 1.0
    else:
        try:
            _, p_value = stats.wilcoxon(gaps, alternative="greater")
        except Exception:
            p_value = 1.0
    
    # Interpretation
    if acc >= 0.65 and p_value < 0.05:
        interp = "strongly valid"
    elif acc >= 0.55 and p_value < 0.05:
        interp = "weakly valid"
    elif acc >= 0.50 and p_value < 0.05:
        interp = "marginal"
    else:
        interp = "invalid"
    
    return acc, mean_gap, cohen_d, p_value, interp


# ── BLOCK 3: ANALYSIS 1 — OVERALL VALIDITY (TYPE A) ─────────────────────────
# What: Run validity on all TYPE A pairs, one head at a time.
# Why: Direct comparison to hh-rlhf. If HelpSteer2 overall validity
#      looks similar to hh-rlhf, the dataset quality isn't the issue.
#      If it's better, dataset cleanliness matters.

print("\n── Analysis 1: Overall Validity (TYPE A pairs) ────────────────────────")
print(f"  N = {len(type_a_indices)} pairs\n")

idx_a = np.array(type_a_indices)
scores_a = head_scores[idx_a]  # (n_a, 2, 19)

print(f"  {'Head':25s} {'Acc':>6s} {'Gap':>8s} {'d':>7s} {'p':>10s}  Interpretation")
print("  " + "-" * 75)

results_overall = {}
for j, head in enumerate(HEAD_NAMES):
    chosen   = scores_a[:, 0, j]
    rejected = scores_a[:, 1, j]
    acc, gap, d, p, interp = compute_validity(chosen, rejected)
    results_overall[head] = {"acc": acc, "gap": gap, "d": d, "p": p, "interp": interp}
    p_str = f"{p:.4f}" if p >= 0.0001 else "<0.0001"
    print(f"  {head:25s} {acc:.3f}  {gap:+.4f}  {d:+.3f}  {p_str:>10s}  {interp}")


# ── BLOCK 4: ANALYSIS 2 — DIMENSION-TARGETED VALIDITY (TYPE B) ──────────────
# What: For each of 5 dimensions, run validity only on pairs where
#       that dimension was the controlled variable.
#       Focus specifically on whether the MATCHING head is valid.
# Why: This is the novel test. On a pair where helpfulness differs by 2+
#      and everything else is controlled — does the helpfulness head agree?
#      We report: (a) the matching head, and (b) all 19 heads, to see
#      whether non-matching heads also activate (entanglement evidence).

print("\n── Analysis 2: Dimension-Targeted Validity (TYPE B pairs) ─────────────")

results_targeted = {}
for target_dim in HS2_DIMS:
    idx_b = np.array(type_b_indices[target_dim])
    if len(idx_b) == 0:
        print(f"\n  [{target_dim}] — no pairs found, skipping")
        continue
    
    scores_b = head_scores[idx_b]  # (n_b, 2, 19)
    n_b = len(idx_b)
    
    print(f"\n  Target dimension: {target_dim.upper()} (N={n_b} pairs)")
    print(f"  {'Head':25s} {'Acc':>6s} {'d':>7s}  Interpretation  {'← MATCH' if True else ''}")
    print("  " + "-" * 65)
    
    results_targeted[target_dim] = {}
    for j, head in enumerate(HEAD_NAMES):
        chosen   = scores_b[:, 0, j]
        rejected = scores_b[:, 1, j]
        acc, gap, d, p, interp = compute_validity(chosen, rejected)
        results_targeted[target_dim][head] = {"acc": acc, "gap": gap, "d": d, "p": p, "interp": interp}
        
        match_marker = "  ← MATCHING HEAD" if head == target_dim else ""
        print(f"  {head:25s} {acc:.3f}  {d:+.3f}  {interp}{match_marker}")


# ── BLOCK 5: THE KEY COMPARISON TABLE ────────────────────────────────────────
# What: For each of the 5 overlapping dimensions, show:
#       hh-rlhf accuracy | HelpSteer2 overall accuracy | HelpSteer2 targeted accuracy
# Why: This is the table that answers the core question.
#      If targeted accuracy > overall > hh-rlhf, the heads work when given
#      clean signal — dataset quality was the problem.
#      If targeted accuracy is still low — the label doesn't match what's learned.

# Load hh-rlhf results for comparison
HHRLHF_RESULTS_PATH = "outputs/validity_results.json"
hhrlhf_results = None
if os.path.exists(HHRLHF_RESULTS_PATH):
    with open(HHRLHF_RESULTS_PATH) as f:
        hhrlhf_results = json.load(f)
    print("\n── Key Comparison: hh-rlhf vs HelpSteer2 (matching heads only) ─────────")
    print(f"\n  {'Dimension':20s} {'hh-rlhf acc':>12s} {'HS2 overall':>12s} {'HS2 targeted':>13s}  Trend")
    print("  " + "-" * 70)
    
    for dim in HS2_DIMS:
        hh_acc = hhrlhf_results.get(dim, {}).get("acc", float("nan"))
        hs2_overall = results_overall.get(dim, {}).get("acc", float("nan"))
        hs2_targeted = results_targeted.get(dim, {}).get(dim, {}).get("acc", float("nan"))
        
        # Trend: are things getting better as signal gets cleaner?
        if hs2_targeted > hs2_overall > hh_acc:
            trend = "↑ signal quality matters"
        elif hs2_targeted <= hh_acc:
            trend = "✗ no recovery — model problem"
        else:
            trend = "~ partial recovery"
        
        print(f"  {dim:20s} {hh_acc:>12.3f} {hs2_overall:>12.3f} {hs2_targeted:>13.3f}  {trend}")
else:
    print(f"\n  (hh-rlhf results not found at {HHRLHF_RESULTS_PATH} — skipping comparison)")
    print("  Run validity_check.py on hh-rlhf outputs first for the full comparison.")


# ── BLOCK 6: SAVE RESULTS ────────────────────────────────────────────────────

out = {
    "overall": results_overall,
    "targeted": results_targeted
}
out_path = os.path.join(INPUT_DIR, "validity_results.json")
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)

print(f"\nSaved → {out_path}")
print("\nDone.")
