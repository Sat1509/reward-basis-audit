"""
extract_rewardbench.py
Scores allenai/reward-bench (filtered split, 2958 pairs) with ArmoRM and saves to outputs_rewardbench/.
Key difference from hh-rlhf: Reward Bench has a `subset` column (chat, chat-hard, safety, reasoning)
which lets us stratify validity by what kind of preference each pair was testing.
Output format mirrors hh-rlhf so analysis scripts can be pointed at either directory.
"""

import numpy as np
import json
import os
import torch
from datasets import load_dataset
from tqdm import tqdm

import sys
sys.path.insert(0, '.')
from utils import load_model_and_tokenizer, score_text

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


# ---
# subset column values: chat, chat_hard, safety, reasoning — preserved in metadata
# for stratified validity analysis in validity_check_rewardbench.py.

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


# ---
# Reward Bench stores prompt/response as separate strings (unlike hh-rlhf's multi-turn format).
# We still need apply_chat_template — gating network looks for token pattern
# [128009, 128006, 78191, 128007, 271] that only appears with the template.

def format_and_score(tokenizer, model, prompt, response, device):
    """Applies chat template and scores a single (prompt, response) pair."""
    messages = [
        {"role": "user",      "content": prompt},
        {"role": "assistant", "content": response},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )

    head_scores, hidden_state, gating_output = score_text(
        model, tokenizer, text, device
    )

    return head_scores, hidden_state, gating_output


# ---
# axis 1: [chosen=0, rejected=1] — same convention as hh-rlhf outputs

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


# ---
# same sanity check as hh-rlhf: most heads should show chosen > rejected if tokenization is correct

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

    subsets = [p['subset'] for p in pairs]
    unique_subsets = sorted(set(subsets))
    print(f"\nPair counts per subset:")
    for s in unique_subsets:
        count = subsets.count(s)
        print(f"  {s:<40} {count} pairs")


# ---

def main():
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
