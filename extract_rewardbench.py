"""
extract_rewardbench.py

Runs ArmoRM on Reward Bench (allenai/reward-bench, filtered split, 2958 pairs)
and saves head scores + hidden states to outputs_rewardbench/.

Reward Bench differs from hh-rlhf in two important ways:
  1. It has a `subset` column — each pair is categorized (chat, chat-hard,
     safety, reasoning, etc.). This lets us run validity checks per category.
  2. chosen/rejected are plain strings, not conversation-formatted lists.
     We still need to apply ArmoRM's chat template — done in score_rb_pair().

Outputs mirror the hh-rlhf outputs structure so existing analysis scripts
can be pointed at outputs_rewardbench/ with minimal changes:
  outputs_rewardbench/head_scores.npy     (N, 2, 19)
  outputs_rewardbench/hidden_states.npy   (N, 2, 4096)
  outputs_rewardbench/gating_outputs.npy  (N, 2, 19)
  outputs_rewardbench/metadata.json       — includes subset labels per pair

Reuses load_model_and_tokenizer() and score_text() from utils.py.
"""

import numpy as np
import json
import os
import torch
from datasets import load_dataset
from tqdm import tqdm

# Reuse existing utils
import sys
sys.path.insert(0, '.')
from utils import load_model_and_tokenizer, score_text

# ── Config ────────────────────────────────────────────────────────────────────

OUTPUT_DIR   = 'outputs_rewardbench'
DATASET_ID   = 'allenai/reward-bench'
SPLIT        = 'filtered'       # 2958 pairs, clean labels
MAX_PAIRS    = None             # set to e.g. 800 to cap; None = use all

HEAD_NAMES = [
    'helpfulness', 'correctness', 'coherence', 'complexity', 'verbosity',
    'safety', 'instruction_following', 'honesty', 'truthfulness', 'harmlessness',
    'readability', 'depth', 'creativity', 'detail', 'positivity', 'clarity',
    'engagement', 'conciseness', 'relevance'
]

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# BLOCK 1: Load Reward Bench
# ══════════════════════════════════════════════════════════════════════════════
# Reward Bench filtered split has 2958 pairs.
# Columns: prompt, chosen, rejected, chosen_model, rejected_model, subset, id
#
# The subset column is the key addition over hh-rlhf. It tells us what kind
# of preference each pair is testing:
#   chat          — general helpfulness (alpacaeval, mt-bench easy/medium)
#   chat_hard     — harder chat (mt-bench-hard, llmbar adversarial)
#   safety        — safety-relevant pairs (specifically constructed)
#   reasoning     — math, coding, logical reasoning
#
# We preserve subset labels in metadata so validity_check_rewardbench.py
# can stratify results by category.

def load_rewardbench(max_pairs=None):
    print(f"Loading {DATASET_ID} ({SPLIT} split) ...")
    ds = load_dataset(DATASET_ID, split=SPLIT)

    if max_pairs is not None:
        ds = ds.select(range(min(max_pairs, len(ds))))

    print(f"  {len(ds)} pairs loaded")
    print(f"  Subsets: {sorted(set(ds['subset']))}")

    pairs = []
    for row in ds:
        pairs.append({
            'prompt'   : row['prompt'],
            'chosen'   : row['chosen'],
            'rejected' : row['rejected'],
            'subset'   : row['subset'],
            'id'       : row['id'],
        })

    return pairs


# ══════════════════════════════════════════════════════════════════════════════
# BLOCK 2: Format Reward Bench pairs for ArmoRM
# ══════════════════════════════════════════════════════════════════════════════
# hh-rlhf stores conversations as multi-turn strings with \n\nHuman:/\n\nAssistant: markers.
# utils.py's tokenize_text() handles that format specifically.
#
# Reward Bench stores prompt and response as separate plain strings.
# ArmoRM still needs its specific chat template applied — the gating network
# looks for a specific token sequence [128009, 128006, 78191, 128007, 271]
# that only appears when apply_chat_template is used correctly.
#
# We format each pair as a single-turn conversation:
#   [{"role": "user",      "content": prompt},
#    {"role": "assistant", "content": response}]
# then apply the tokenizer's chat template — same as utils.py does internally.

def format_and_score(tokenizer, model, prompt, response, device):
    """
    Format a (prompt, response) pair into ArmoRM's expected input and score it.
    Returns (head_scores, hidden_state, gating_output) as numpy arrays.
    """
    messages = [
        {"role": "user",      "content": prompt},
        {"role": "assistant", "content": response},
    ]

    # apply_chat_template produces the exact token sequence ArmoRM expects
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )

    # Reuse score_text from utils — it handles tokenization + forward pass
    # score_text expects a pre-formatted string, which we now have
    head_scores, hidden_state, gating_output = score_text(
        model, tokenizer, text, device
    )

    return head_scores, hidden_state, gating_output


# ══════════════════════════════════════════════════════════════════════════════
# BLOCK 3: Score all pairs and save
# ══════════════════════════════════════════════════════════════════════════════
# Same structure as extract_scores_and_embeddings.py:
#   head_scores  (N, 2, 19) — axis 1: [chosen=0, rejected=1]
#   hidden_states (N, 2, 4096)
#   gating_outputs (N, 2, 19)
#   metadata.json — includes subset labels, pair ids

def score_all_pairs(pairs, model, tokenizer, device):
    N = len(pairs)
    head_scores   = np.zeros((N, 2, 19), dtype=np.float32)
    hidden_states = np.zeros((N, 2, 4096), dtype=np.float32)
    gating_outputs = np.zeros((N, 2, 19), dtype=np.float32)

    print(f"\nScoring {N} pairs ...")
    for i, pair in enumerate(tqdm(pairs)):
        for j, response_key in enumerate(['chosen', 'rejected']):
            hs, hid, gate = format_and_score(
                tokenizer, model,
                pair['prompt'], pair[response_key],
                device
            )
            head_scores[i, j]    = hs
            hidden_states[i, j]  = hid
            gating_outputs[i, j] = gate

    return head_scores, hidden_states, gating_outputs


def save_outputs(pairs, head_scores, hidden_states, gating_outputs):
    np.save(f'{OUTPUT_DIR}/head_scores.npy',    head_scores)
    np.save(f'{OUTPUT_DIR}/hidden_states.npy',  hidden_states)
    np.save(f'{OUTPUT_DIR}/gating_outputs.npy', gating_outputs)

    metadata = {
        'n_pairs'    : len(pairs),
        'head_names' : HEAD_NAMES,
        'dataset'    : DATASET_ID,
        'split'      : SPLIT,
        'subsets'    : [p['subset'] for p in pairs],
        'ids'        : [p['id'] for p in pairs],
    }
    with open(f'{OUTPUT_DIR}/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved head_scores    → {OUTPUT_DIR}/head_scores.npy    shape: {head_scores.shape}")
    print(f"Saved hidden_states  → {OUTPUT_DIR}/hidden_states.npy  shape: {hidden_states.shape}")
    print(f"Saved gating_outputs → {OUTPUT_DIR}/gating_outputs.npy shape: {gating_outputs.shape}")
    print(f"Saved metadata       → {OUTPUT_DIR}/metadata.json")


# ══════════════════════════════════════════════════════════════════════════════
# BLOCK 4: Sanity check
# ══════════════════════════════════════════════════════════════════════════════
# Same check as hh-rlhf run: mean score gap (chosen - rejected) per head.
# Should be positive for most heads if tokenization is correct.
# Also print per-subset pair counts so we know the breakdown.

def sanity_check(pairs, head_scores):
    gaps = head_scores[:, 0, :] - head_scores[:, 1, :]   # chosen - rejected
    mean_gaps = gaps.mean(axis=0)

    n_positive = (mean_gaps > 0).sum()
    print(f"\nMean score gap (chosen - rejected) per head:")
    print(f"  {'Head':<28} {'Gap':>8}  Direction")
    print(f"  {'-'*50}")
    for i, name in enumerate(HEAD_NAMES):
        direction = '✓ chosen higher' if mean_gaps[i] > 0 else '✗ rejected higher'
        print(f"  {name:<28} {mean_gaps[i]:>+8.4f}  {direction}')
    print(f"\n{n_positive}/19 heads score chosen > rejected on average.")

    # Subset breakdown
    subsets = [p['subset'] for p in pairs]
    unique_subsets = sorted(set(subsets))
    print(f"\nPair counts per subset:")
    for s in unique_subsets:
        count = subsets.count(s)
        print(f"  {s:<40} {count} pairs")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # Guard: skip if outputs already exist
    if os.path.exists(f'{OUTPUT_DIR}/head_scores.npy'):
        print(f"Outputs already exist in {OUTPUT_DIR}/. Delete to re-run.")
        return

    pairs = load_rewardbench(max_pairs=MAX_PAIRS)

    print("\nLoading model ...")
    model, tokenizer = load_model_and_tokenizer()
    device = next(model.parameters()).device

    head_scores, hidden_states, gating_outputs = score_all_pairs(
        pairs, model, tokenizer, device
    )

    sanity_check(pairs, head_scores)
    save_outputs(pairs, head_scores, hidden_states, gating_outputs)

    print("\nDone. Run validity_check_rewardbench.py next.")


if __name__ == '__main__':
    main()
