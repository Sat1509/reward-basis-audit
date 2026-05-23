"""
extract_helpsteer2.py

Runs ArmoRM on HelpSteer2 and saves head scores + hidden states.

HelpSteer2 structure:
  - Each row: (prompt, response, helpfulness, correctness, coherence, complexity, verbosity)
  - Multiple responses per prompt exist (different rows, same prompt)
  - Ratings are integers 0-4 (Likert scale)

We construct TWO kinds of pairs:

  TYPE A — Overall pairs:
    For each prompt with 2+ responses, pick the pair with the
    largest total rating gap (sum of all 5 dimensions).
    "chosen" = higher total, "rejected" = lower total.
    This mirrors hh-rlhf logic for a direct comparison.

  TYPE B — Dimension-targeted pairs:
    For each of the 5 dimensions, build pairs where:
      - That dimension has a gap of >= 2
      - All other dimensions differ by <= 1
    This lets us ask: when helpfulness is the primary difference,
    does the helpfulness head agree?
    This is the test ArmoRM's paper never ran.

Why two types?
  Type A = apples-to-apples comparison with hh-rlhf results.
  Type B = the targeted test that uniquely exploits HelpSteer2's structure.
  Type B is the novel finding. Type A is the sanity check.
"""

import json
import numpy as np
from datasets import load_dataset
from collections import defaultdict

# ── CONFIG ──────────────────────────────────────────────────────────────────

DATASET_NAME = "nvidia/HelpSteer2"
SPLIT = "train"
OUTPUT_DIR = "outputs_helpsteer2"

# ArmoRM head names (same order as the model outputs)
HEAD_NAMES = [
    "helpfulness", "correctness", "coherence", "complexity", "verbosity",
    "safety", "instruction_following", "honesty", "truthfulness", "harmlessness",
    "readability", "depth", "creativity", "detail", "positivity",
    "clarity", "engagement", "conciseness", "relevance"
]

# HelpSteer2 dimension names (subset of head names — these 5 overlap exactly)
HS2_DIMS = ["helpfulness", "correctness", "coherence", "complexity", "verbosity"]

# Pair construction thresholds for TYPE B
TARGET_DIM_GAP = 2      # target dimension must differ by at least this
CONTROL_DIM_GAP = 1     # other dimensions must differ by at most this
MAX_PAIRS_PER_DIM = 200 # cap per dimension to keep compute manageable

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── BLOCK 1: LOAD AND GROUP BY PROMPT ───────────────────────────────────────
# What: Load HelpSteer2, group all responses by prompt text.
# Why: We need multiple responses to the same prompt to form pairs.
#      Without grouping, we can't compare two responses to the same question.
# Output: dict of {prompt_text: [list of response dicts]}

print("Loading HelpSteer2 ...")
ds = load_dataset(DATASET_NAME, split=SPLIT)
print(f"  {len(ds)} rows loaded")

# Group rows by prompt
grouped = defaultdict(list)
for row in ds:
    grouped[row["prompt"]].append({
        "response": row["response"],
        "helpfulness": row["helpfulness"],
        "correctness": row["correctness"],
        "coherence": row["coherence"],
        "complexity": row["complexity"],
        "verbosity": row["verbosity"],
        "total": sum([row["helpfulness"], row["correctness"],
                      row["coherence"], row["complexity"], row["verbosity"]])
    })

multi_prompts = {p: resps for p, resps in grouped.items() if len(resps) >= 2}
print(f"  {len(multi_prompts)} prompts with 2+ responses")


# ── BLOCK 2: BUILD TYPE A PAIRS (overall) ───────────────────────────────────
# What: For each prompt, find the pair of responses with the largest
#       total rating gap. chosen = higher total, rejected = lower.
# Why: Mirrors hh-rlhf structure — overall preference, no dimension control.
#      Lets us directly compare HelpSteer2 validity results to hh-rlhf results.
# Output: list of (prompt, chosen_response, rejected_response, metadata)

print("\nBuilding TYPE A pairs (overall preference) ...")
type_a_pairs = []

for prompt, responses in multi_prompts.items():
    # Sort by total score descending
    sorted_resps = sorted(responses, key=lambda x: x["total"], reverse=True)
    best = sorted_resps[0]
    worst = sorted_resps[-1]
    gap = best["total"] - worst["total"]
    
    if gap > 0:  # only keep pairs where there's actually a difference
        type_a_pairs.append({
            "prompt": prompt,
            "chosen_response": best["response"],
            "rejected_response": worst["response"],
            "chosen_ratings": {d: best[d] for d in HS2_DIMS},
            "rejected_ratings": {d: worst[d] for d in HS2_DIMS},
            "total_gap": gap,
            "pair_type": "overall"
        })

print(f"  {len(type_a_pairs)} TYPE A pairs constructed")


# ── BLOCK 3: BUILD TYPE B PAIRS (dimension-targeted) ────────────────────────
# What: For each of 5 dimensions, find pairs where that dimension has a
#       large gap (>=2) but other dimensions are similar (gap <=1).
# Why: This is the core novel test. If we find a pair where response A
#      scores 4 on helpfulness and response B scores 1, but they're equal
#      on everything else — does ArmoRM's helpfulness head agree?
#      This isolates one dimension, which hh-rlhf cannot do.
# Output: dict of {dimension: [list of targeted pairs]}

print("\nBuilding TYPE B pairs (dimension-targeted) ...")
type_b_pairs = defaultdict(list)

for target_dim in HS2_DIMS:
    other_dims = [d for d in HS2_DIMS if d != target_dim]
    
    for prompt, responses in multi_prompts.items():
        # Try all pairs of responses for this prompt
        for i in range(len(responses)):
            for j in range(i + 1, len(responses)):
                r1, r2 = responses[i], responses[j]
                
                target_gap = abs(r1[target_dim] - r2[target_dim])
                other_gaps = [abs(r1[d] - r2[d]) for d in other_dims]
                
                # Check criteria
                if target_gap >= TARGET_DIM_GAP and all(g <= CONTROL_DIM_GAP for g in other_gaps):
                    # chosen = higher on target dimension
                    chosen = r1 if r1[target_dim] > r2[target_dim] else r2
                    rejected = r2 if r1[target_dim] > r2[target_dim] else r1
                    
                    type_b_pairs[target_dim].append({
                        "prompt": prompt,
                        "chosen_response": chosen["response"],
                        "rejected_response": rejected["response"],
                        "chosen_ratings": {d: chosen[d] for d in HS2_DIMS},
                        "rejected_ratings": {d: rejected[d] for d in HS2_DIMS},
                        "target_dim": target_dim,
                        "target_gap": target_gap,
                        "other_gaps": dict(zip(other_dims, other_gaps)),
                        "pair_type": f"targeted_{target_dim}"
                    })
    
    # Cap to avoid runaway compute
    if len(type_b_pairs[target_dim]) > MAX_PAIRS_PER_DIM:
        # Sort by target gap descending (cleaner signal first) then cap
        type_b_pairs[target_dim].sort(key=lambda x: x["target_gap"], reverse=True)
        type_b_pairs[target_dim] = type_b_pairs[target_dim][:MAX_PAIRS_PER_DIM]
    
    print(f"  {target_dim:20s}: {len(type_b_pairs[target_dim])} targeted pairs")

total_b = sum(len(v) for v in type_b_pairs.values())
print(f"  Total TYPE B pairs: {total_b}")


# ── BLOCK 4: FLATTEN ALL PAIRS FOR SCORING ──────────────────────────────────
# What: Combine all pairs into one flat list for ArmoRM to score.
# Why: We run ArmoRM once on everything. Scoring is the expensive step
#      (GPU time). We save all scores, then split by pair type for analysis.
# Output: flat list of all pairs, with type/dim metadata preserved

all_pairs = type_a_pairs.copy()
for dim, pairs in type_b_pairs.items():
    all_pairs.extend(pairs)

print(f"\nTotal pairs to score: {len(all_pairs)}")
print(f"  TYPE A (overall): {len(type_a_pairs)}")
print(f"  TYPE B (targeted): {total_b}")


# ── BLOCK 5: SAVE PAIR METADATA ──────────────────────────────────────────────
# What: Save the pair list (without scores) so we can reconstruct
#       which pair is which after scoring.
# Why: ArmoRM outputs arrays indexed by position. We need the metadata
#      to map position → pair type → dimension → analysis.

metadata_path = os.path.join(OUTPUT_DIR, "metadata.json")
with open(metadata_path, "w") as f:
    json.dump({
        "dataset": DATASET_NAME,
        "split": SPLIT,
        "n_pairs_total": len(all_pairs),
        "n_type_a": len(type_a_pairs),
        "n_type_b": total_b,
        "type_b_counts": {d: len(type_b_pairs[d]) for d in HS2_DIMS},
        "head_names": HEAD_NAMES,
        "hs2_dims": HS2_DIMS,
        "pairs": all_pairs
    }, f, indent=2)

print(f"\nMetadata saved → {metadata_path}")
print("Next: run score_helpsteer2.py to run ArmoRM on these pairs")
print("(Scoring is the GPU step — run on Kaggle/Colab)")
