"""
extract_helpsteer2.py
Builds chosen/rejected pairs from HelpSteer2 and saves pair metadata for scoring.

TYPE A — overall pairs: highest vs lowest total rating per prompt. Direct hh-rlhf comparison.
TYPE B — dimension-targeted pairs: one dimension differs by >=2, all others by <=1.
Type B is the novel test — when helpfulness is the isolated variable, does the helpfulness head agree?
This is the test ArmoRM's paper never ran.
"""

import json
import numpy as np
from datasets import load_dataset
from collections import defaultdict

DATASET_NAME = "nvidia/HelpSteer2"
SPLIT = "train"
OUTPUT_DIR = "outputs_helpsteer2"

# same order as model outputs — 19 heads
HEAD_NAMES = [
    "helpfulness", "correctness", "coherence", "complexity", "verbosity",
    "safety", "instruction_following", "honesty", "truthfulness", "harmlessness",
    "readability", "depth", "creativity", "detail", "positivity",
    "clarity", "engagement", "conciseness", "relevance"
]

# these 5 overlap exactly with HelpSteer2's annotation dimensions
HS2_DIMS = ["helpfulness", "correctness", "coherence", "complexity", "verbosity"]

# TYPE B thresholds
TARGET_DIM_GAP = 2      # target dimension must differ by at least this
CONTROL_DIM_GAP = 1     # other dimensions must differ by at most this
MAX_PAIRS_PER_DIM = 200 # cap per dimension to keep compute manageable

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---

print("Loading HelpSteer2 ...")
ds = load_dataset(DATASET_NAME, split=SPLIT)
print(f"  {len(ds)} rows loaded")

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


# ---
# TYPE A: highest vs lowest total rating per prompt. Mirrors hh-rlhf for direct comparison.

print("\nBuilding TYPE A pairs (overall preference) ...")
type_a_pairs = []

for prompt, responses in multi_prompts.items():
    sorted_resps = sorted(responses, key=lambda x: x["total"], reverse=True)
    best = sorted_resps[0]
    worst = sorted_resps[-1]
    gap = best["total"] - worst["total"]
    
    if gap > 0:
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


# ---
# TYPE B: isolate one dimension per pair. If helpfulness differs by >=2 but everything else
# differs by <=1, does the helpfulness head correctly rank chosen above rejected?

print("\nBuilding TYPE B pairs (dimension-targeted) ...")
type_b_pairs = defaultdict(list)

for target_dim in HS2_DIMS:
    other_dims = [d for d in HS2_DIMS if d != target_dim]

    for prompt, responses in multi_prompts.items():
        for i in range(len(responses)):
            for j in range(i + 1, len(responses)):
                r1, r2 = responses[i], responses[j]

                target_gap = abs(r1[target_dim] - r2[target_dim])
                other_gaps = [abs(r1[d] - r2[d]) for d in other_dims]

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
    
    if len(type_b_pairs[target_dim]) > MAX_PAIRS_PER_DIM:
        # strongest signal first, then cap
        type_b_pairs[target_dim].sort(key=lambda x: x["target_gap"], reverse=True)
        type_b_pairs[target_dim] = type_b_pairs[target_dim][:MAX_PAIRS_PER_DIM]
    
    print(f"  {target_dim:20s}: {len(type_b_pairs[target_dim])} targeted pairs")

total_b = sum(len(v) for v in type_b_pairs.values())
print(f"  Total TYPE B pairs: {total_b}")


# ---
# flatten TYPE A + TYPE B into one list — score once, split by pair_type later

all_pairs = type_a_pairs.copy()
for dim, pairs in type_b_pairs.items():
    all_pairs.extend(pairs)

print(f"\nTotal pairs to score: {len(all_pairs)}")
print(f"  TYPE A (overall): {len(type_a_pairs)}")
print(f"  TYPE B (targeted): {total_b}")


# ---
# ArmoRM outputs arrays indexed by position — metadata maps position → pair_type → dimension.

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
