"""
validity_check_rewardbench.py

Runs the same validity analysis as validity_check.py but on Reward Bench outputs,
with one important addition: results are stratified by subset.

This directly addresses the main limitation of the hh-rlhf validity check:
  - hh-rlhf pairs vary on many dimensions simultaneously (quality, safety,
    helpfulness all at once), making it hard to isolate which concept drove preference.
  - Reward Bench subsets are constructed to test specific capabilities.
    The 'safety' subset specifically tests safety-relevant preference judgments.
    The 'reasoning' subset tests reasoning/correctness.

So the key question this script answers:
  Does the safety head fail on the Reward Bench SAFETY subset specifically?

If yes: the safety head fails even on pairs where safety was the explicit
selection criterion. This is a much stronger claim than the hh-rlhf result.

If no: the failure is dataset-specific. The hh-rlhf safety finding doesn't
generalize. That's also important to report honestly.

Outputs:
  outputs_rewardbench/validity_results.json     — full results per head
  outputs_rewardbench/validity_by_subset.json   — results per head per subset
  outputs_rewardbench/validity_comparison.json  — side-by-side with hh-rlhf results
  outputs_rewardbench/validity_bar.png
  outputs_rewardbench/validity_by_subset.png
"""

import numpy as np
import json
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

OUTPUT_DIR   = 'outputs_rewardbench'
HHRXLHF_DIR = 'outputs'           # original hh-rlhf results for comparison

HEAD_NAMES = [
    'helpfulness', 'correctness', 'coherence', 'complexity', 'verbosity',
    'safety', 'instruction_following', 'honesty', 'truthfulness', 'harmlessness',
    'readability', 'depth', 'creativity', 'detail', 'positivity', 'clarity',
    'engagement', 'conciseness', 'relevance'
]
N_HEADS = len(HEAD_NAMES)

SAFETY_IDX     = HEAD_NAMES.index('safety')
READABILITY_IDX = HEAD_NAMES.index('readability')


# ══════════════════════════════════════════════════════════════════════════════
# BLOCK 1: Validity metrics — same as validity_check.py
# ══════════════════════════════════════════════════════════════════════════════
# For each head:
#   gap[i] = chosen_score[i] - rejected_score[i]  for each pair i
#   preference_accuracy = fraction of pairs where gap > 0
#   Wilcoxon signed-rank test on gaps (tests whether median gap ≠ 0)
#   Cohen's d = mean(gap) / std(gap)
#
# Same thresholds as before:
#   strongly valid  : acc ≥ 0.65, p < 0.05
#   weakly valid    : acc ≥ 0.55, p < 0.05
#   invalid         : everything else

def compute_validity(head_scores):
    """
    Args:
        head_scores : (N, 2, 19) — axis 1: [chosen=0, rejected=1]

    Returns:
        results : list of dicts, one per head
    """
    chosen   = head_scores[:, 0, :]    # (N, 19)
    rejected = head_scores[:, 1, :]    # (N, 19)
    gaps     = chosen - rejected        # (N, 19)
    N = len(gaps)

    results = []
    for i, name in enumerate(HEAD_NAMES):
        g = gaps[:, i]
        pref_acc = float((g > 0).mean())
        mean_gap = float(g.mean())
        std_gap  = float(g.std())
        cohens_d = mean_gap / std_gap if std_gap > 0 else 0.0

        _, p_val = stats.wilcoxon(g, alternative='greater')

        if pref_acc >= 0.65 and p_val < 0.05:
            interpretation = 'strongly valid'
        elif pref_acc >= 0.55 and p_val < 0.05:
            interpretation = 'weakly valid'
        elif pref_acc >= 0.50 and p_val < 0.05:
            interpretation = 'invalid — marginal positive'
        elif pref_acc >= 0.45:
            interpretation = 'invalid — chance'
        else:
            interpretation = 'invalid — inverted'

        results.append({
            'head'            : name,
            'n_pairs'         : N,
            'pref_acc'        : pref_acc,
            'mean_gap'        : mean_gap,
            'cohens_d'        : cohens_d,
            'p_value'         : float(p_val),
            'interpretation'  : interpretation,
        })

    return results


# ══════════════════════════════════════════════════════════════════════════════
# BLOCK 2: Per-subset validity
# ══════════════════════════════════════════════════════════════════════════════
# Reward Bench's subset column lets us ask targeted questions:
#
#   On pairs where SAFETY was the selection criterion,
#   does the safety head predict the preference?
#
#   On pairs where REASONING was the selection criterion,
#   does the correctness/depth head predict the preference?
#
# This is a much cleaner test than overall validity because each subset
# controls for what the human rater was evaluating.
#
# We run compute_validity() separately on the rows belonging to each subset.

def compute_validity_by_subset(head_scores, subsets):
    """
    Args:
        head_scores : (N, 2, 19)
        subsets     : list of N subset labels

    Returns:
        by_subset : dict mapping subset_name → validity results list
    """
    subsets_arr = np.array(subsets)
    unique = sorted(set(subsets))
    by_subset = {}

    for s in unique:
        mask = subsets_arr == s
        subset_scores = head_scores[mask]
        if len(subset_scores) < 10:
            continue   # too few pairs for meaningful statistics
        by_subset[s] = compute_validity(subset_scores)
        print(f"  Subset '{s}': {mask.sum()} pairs")

    return by_subset


# ══════════════════════════════════════════════════════════════════════════════
# BLOCK 3: Comparison with hh-rlhf
# ══════════════════════════════════════════════════════════════════════════════
# Load hh-rlhf validity results if available (from outputs/validity_results.json)
# and build a side-by-side comparison table.
#
# The comparison tells us whether our findings are:
#   Consistent across datasets  → property of the model
#   Dataset-specific            → property of hh-rlhf's construction
#
# For the safety head specifically, we want to compare:
#   hh-rlhf overall: 0.520
#   Reward Bench overall: ?
#   Reward Bench safety subset: ?    ← most important number

def load_hhrxlhf_results():
    path = f'{HHRXLHF_DIR}/validity_results.json'
    if not os.path.exists(path):
        print(f"  hh-rlhf validity results not found at {path}. Skipping comparison.")
        return None
    with open(path) as f:
        return json.load(f)


def build_comparison(rb_results, hh_results):
    """
    Returns a list of dicts with pref_acc from both datasets per head.
    """
    if hh_results is None:
        return None

    # hh-rlhf results might be a dict or list depending on how validity_check.py saved them
    if isinstance(hh_results, dict):
        hh_by_head = hh_results
    else:
        hh_by_head = {r['head']: r for r in hh_results}

    rb_by_head = {r['head']: r for r in rb_results}

    comparison = []
    for name in HEAD_NAMES:
        hh = hh_by_head.get(name, {})
        rb = rb_by_head.get(name, {})
        comparison.append({
            'head'          : name,
            'hhrxlhf_acc'   : hh.get('pref_acc', None),
            'rewardbench_acc': rb.get('pref_acc', None),
            'consistent'    : abs(
                (hh.get('pref_acc', 0.5) - 0.5) -
                (rb.get('pref_acc', 0.5) - 0.5)
            ) < 0.10   # within 10pp = consistent direction
        })

    return comparison


# ══════════════════════════════════════════════════════════════════════════════
# PRINTING
# ══════════════════════════════════════════════════════════════════════════════

def print_results(results, label=""):
    n = results[0]['n_pairs']
    print(f"\n── Validity Results {label} (N={n}) ────────────────────────────────────────")
    print(f"  {'Head':<26} {'Acc':>6}  {'Gap':>8}  {'d':>6}  {'p':>8}  Interpretation")
    print(f"  {'-'*85}")
    for r in results:
        print(f"  {r['head']:<26} {r['pref_acc']:>6.3f}  {r['mean_gap']:>+8.4f}  "
              f"{r['cohens_d']:>+6.3f}  {r['p_value']:>8.4f}  {r['interpretation']}")

    valid    = [r['head'] for r in results if 'strongly' in r['interpretation']]
    marginal = [r['head'] for r in results if 'weakly'   in r['interpretation']]
    invalid  = [r['head'] for r in results
                if 'strongly' not in r['interpretation']
                and 'weakly'  not in r['interpretation']]

    print(f"\n  Valid    (acc≥0.65, p<0.05) : {valid}")
    print(f"  Marginal (acc≥0.55, p<0.05) : {marginal}")
    print(f"  Invalid                      : {len(invalid)} heads")

    safety = next(r for r in results if r['head'] == 'safety')
    print(f"\n  Safety head: acc={safety['pref_acc']:.3f}  d={safety['cohens_d']:+.3f}  "
          f"p={safety['p_value']:.4f}  → {safety['interpretation']}")


def print_subset_safety(by_subset):
    """Print safety head results for each subset."""
    print(f"\n── Safety head preference accuracy by subset ───────────────────────────────")
    print(f"  {'Subset':<40} {'N':>5}  {'Acc':>6}  {'p':>8}  Interpretation")
    print(f"  {'-'*75}")
    for subset, results in sorted(by_subset.items()):
        safety = next(r for r in results if r['head'] == 'safety')
        print(f"  {subset:<40} {safety['n_pairs']:>5}  {safety['pref_acc']:>6.3f}  "
              f"{safety['p_value']:>8.4f}  {safety['interpretation']}")


def print_comparison(comparison):
    if comparison is None:
        return
    print(f"\n── Cross-dataset comparison (preference accuracy) ──────────────────────────")
    print(f"  {'Head':<26} {'hh-rlhf':>8}  {'RewBench':>9}  {'Δ':>6}  Consistent?")
    print(f"  {'-'*60}")
    for r in comparison:
        hh  = r['hhrxlhf_acc']
        rb  = r['rewardbench_acc']
        if hh is None or rb is None:
            continue
        delta = rb - hh
        flag  = '✓' if r['consistent'] else '✗'
        print(f"  {r['head']:<26} {hh:>8.3f}  {rb:>9.3f}  {delta:>+6.3f}  {flag}")


# ══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

def plot_results(rb_results, by_subset, comparison):
    fig = plt.figure(figsize=(18, 12))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    accs   = [r['pref_acc'] for r in rb_results]
    colors = ['green' if a >= 0.65 else 'orange' if a >= 0.55 else 'red' for a in accs]

    # Panel 1: Overall Reward Bench validity
    ax1 = fig.add_subplot(gs[0, 0])
    bars = ax1.barh(HEAD_NAMES, accs, color=colors, alpha=0.75)
    ax1.axvline(0.5,  color='black', linestyle='--', linewidth=1, label='Chance (0.50)')
    ax1.axvline(0.65, color='green', linestyle='--', linewidth=1, alpha=0.6, label='Valid (0.65)')
    ax1.set_xlabel('Preference accuracy', fontsize=9)
    ax1.set_title('Reward Bench: Overall Validity\nAll subsets pooled', fontsize=10, fontweight='bold')
    ax1.set_xlim(0.3, 0.85)
    ax1.legend(fontsize=8)

    # Panel 2: Safety head across subsets
    ax2 = fig.add_subplot(gs[0, 1])
    subset_names = sorted(by_subset.keys())
    safety_accs  = [next(r for r in by_subset[s] if r['head'] == 'safety')['pref_acc']
                    for s in subset_names]
    subset_colors = ['green' if a >= 0.65 else 'orange' if a >= 0.55 else 'red'
                     for a in safety_accs]
    ax2.barh(subset_names, safety_accs, color=subset_colors, alpha=0.75)
    ax2.axvline(0.5,  color='black', linestyle='--', linewidth=1)
    ax2.axvline(0.65, color='green', linestyle='--', linewidth=1, alpha=0.6)
    ax2.set_xlabel('Preference accuracy', fontsize=9)
    ax2.set_title("Safety Head by Subset\nDoes it work on safety-specific pairs?",
                  fontsize=10, fontweight='bold')
    ax2.set_xlim(0.3, 0.85)

    # Panel 3: Cross-dataset comparison (if available)
    ax3 = fig.add_subplot(gs[1, :])
    if comparison is not None:
        hh_accs = [r['hhrxlhf_acc']    for r in comparison if r['hhrxlhf_acc'] is not None]
        rb_accs = [r['rewardbench_acc'] for r in comparison if r['rewardbench_acc'] is not None]
        names   = [r['head']            for r in comparison if r['hhrxlhf_acc'] is not None]
        x = np.arange(len(names))
        w = 0.35
        ax3.bar(x - w/2, hh_accs, w, label='hh-rlhf',       color='steelblue', alpha=0.75)
        ax3.bar(x + w/2, rb_accs, w, label='Reward Bench',   color='darkorange', alpha=0.75)
        ax3.axhline(0.5,  color='black', linestyle='--', linewidth=1)
        ax3.axhline(0.65, color='green', linestyle='--', linewidth=1, alpha=0.6)
        ax3.set_xticks(x)
        ax3.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
        ax3.set_ylabel('Preference accuracy', fontsize=9)
        ax3.set_title('Cross-Dataset Validity Comparison\nhh-rlhf vs Reward Bench',
                      fontsize=10, fontweight='bold')
        ax3.legend(fontsize=9)
        ax3.set_ylim(0.3, 0.85)
    else:
        ax3.text(0.5, 0.5, 'hh-rlhf results not found\n(run validity_check.py first)',
                 ha='center', va='center', transform=ax3.transAxes, fontsize=12)

    plt.suptitle('Reward Bench Validity Analysis\nArmoRM reward head preference accuracy',
                 fontsize=13, fontweight='bold')
    out = f'{OUTPUT_DIR}/validity_rewardbench.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved figure → {out}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("Loading Reward Bench outputs ...")
    head_scores = np.load(f'{OUTPUT_DIR}/head_scores.npy')     # (N, 2, 19)
    with open(f'{OUTPUT_DIR}/metadata.json') as f:
        metadata = json.load(f)
    subsets = metadata['subsets']
    N = len(subsets)
    print(f"  {N} pairs, {len(set(subsets))} subsets")

    # ── Overall validity ──────────────────────────────────────────────────────
    print("\nComputing overall validity ...")
    rb_results = compute_validity(head_scores)
    print_results(rb_results, label="— Reward Bench (all subsets)")

    # ── Per-subset validity ───────────────────────────────────────────────────
    print("\nComputing per-subset validity ...")
    by_subset = compute_validity_by_subset(head_scores, subsets)
    print_subset_safety(by_subset)

    # ── Cross-dataset comparison ──────────────────────────────────────────────
    print("\nLoading hh-rlhf results for comparison ...")
    hh_results = load_hhrxlhf_results()
    comparison = build_comparison(rb_results, hh_results)
    print_comparison(comparison)

    # ── Save results ──────────────────────────────────────────────────────────
    with open(f'{OUTPUT_DIR}/validity_results.json', 'w') as f:
        json.dump(rb_results, f, indent=2)

    subset_serializable = {k: v for k, v in by_subset.items()}
    with open(f'{OUTPUT_DIR}/validity_by_subset.json', 'w') as f:
        json.dump(subset_serializable, f, indent=2)

    if comparison:
        with open(f'{OUTPUT_DIR}/validity_comparison.json', 'w') as f:
            json.dump(comparison, f, indent=2)

    print(f"\nSaved → {OUTPUT_DIR}/validity_results.json")
    print(f"Saved → {OUTPUT_DIR}/validity_by_subset.json")
    if comparison:
        print(f"Saved → {OUTPUT_DIR}/validity_comparison.json")

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_results(rb_results, by_subset, comparison)

    # ── Headline ──────────────────────────────────────────────────────────────
    valid    = [r['head'] for r in rb_results if 'strongly' in r['interpretation']]
    safety_r = next(r for r in rb_results if r['head'] == 'safety')

    print(f"\n── Headline ────────────────────────────────────────────────────────────────")
    print(f"  Valid heads on Reward Bench : {valid if valid else 'none'}")
    print(f"  Safety head overall         : {safety_r['pref_acc']:.3f}")

    if 'safety' in by_subset:
        safety_subset_r = next(r for r in by_subset['safety'] if r['head'] == 'safety')
        print(f"  Safety head on safety subset: {safety_subset_r['pref_acc']:.3f}")
        print(f"\n  Key question answered: does the safety head fail specifically")
        print(f"  on pairs where safety was the explicit selection criterion?")
        if safety_subset_r['pref_acc'] < 0.60:
            print(f"  → YES. acc={safety_subset_r['pref_acc']:.3f}. "
                  f"Finding replicates and strengthens.")
        else:
            print(f"  → NO. acc={safety_subset_r['pref_acc']:.3f}. "
                  f"Safety head works better on targeted safety pairs.")
            print(f"     This is an important nuance to report honestly.")


if __name__ == '__main__':
    main()
