"""
score_helpsteer2.py

Runs ArmoRM on the pairs built by extract_helpsteer2.py.
Saves head scores, hidden states, gating outputs — same format as hh-rlhf outputs.

Run this on Kaggle/Colab (GPU required).
Assumes extract_helpsteer2.py has already been run and metadata.json exists.

Output arrays:
  head_scores.npy    shape: (N, 2, 19)  — axis 1: [chosen=0, rejected=1]
  hidden_states.npy  shape: (N, 2, 4096)
  gating_outputs.npy shape: (N, 2, 19)
"""

import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import os

# ── CONFIG ──────────────────────────────────────────────────────────────────

INPUT_DIR = "outputs_helpsteer2"
OUTPUT_DIR = "outputs_helpsteer2"
MODEL_NAME = "RLHFlow/ArmoRM-Llama3-8B-v0.1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

HEAD_NAMES = [
    "helpfulness", "correctness", "coherence", "complexity", "verbosity",
    "safety", "instruction_following", "honesty", "truthfulness", "harmlessness",
    "readability", "depth", "creativity", "detail", "positivity",
    "clarity", "engagement", "conciseness", "relevance"
]


# ── BLOCK 1: LOAD METADATA ──────────────────────────────────────────────────
# What: Load the pair list saved by extract_helpsteer2.py.
# Why: This tells us which (prompt, chosen, rejected) to score.

print("Loading metadata ...")
with open(os.path.join(INPUT_DIR, "metadata.json")) as f:
    meta = json.load(f)

pairs = meta["pairs"]
N = len(pairs)
print(f"  {N} pairs to score")


# ── BLOCK 2: LOAD MODEL ──────────────────────────────────────────────────────
# What: Load ArmoRM in float16 on GPU.
# Why: Same as hh-rlhf pipeline — identical model, identical forward pass.
#      This keeps the comparison fair: same model, different dataset.

print("\nLoading tokenizer ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

print("Loading model (this takes ~1-2 min) ...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    trust_remote_code=True
).to(DEVICE)
model.eval()
print(f"  Model loaded on: {DEVICE}")
print(f"  Num reward heads: {model.config.num_labels}")


# ── BLOCK 3: TOKENIZATION HELPER ────────────────────────────────────────────
# What: Format (prompt, response) into ArmoRM's expected chat template.
# Why: ArmoRM's gating network looks for a specific token pattern to find
#      where the response starts. apply_chat_template is mandatory.
#      Same logic as hh-rlhf pipeline.

def tokenize(prompt, response):
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    tokens = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=4096
    ).to(DEVICE)
    return tokens


# ── BLOCK 4: SCORE ONE RESPONSE ──────────────────────────────────────────────
# What: Single forward pass through ArmoRM.
# Why: Returns 19 head scores, 4096-dim hidden state, 19 gating weights.
#      We save all three for every response (chosen and rejected).
#      head_scores is what we analyze. hidden_states is for probing.
#      gating_outputs tells us how the model weighted each head.

@torch.no_grad()
def score(prompt, response):
    tokens = tokenize(prompt, response)
    outputs = model(**tokens)
    head_scores = outputs.rewards[0].cpu().float().numpy()       # (19,)
    hidden_state = outputs.hidden_states[0].cpu().float().numpy() # (4096,)
    gating = outputs.gating_output[0].cpu().float().numpy()       # (19,)
    return head_scores, hidden_state, gating


# ── BLOCK 5: SCORE ALL PAIRS ─────────────────────────────────────────────────
# What: Run score() on every chosen and rejected response.
# Why: We need both to compute the gap (chosen_score - rejected_score)
#      and check if the head ranks chosen above rejected.
# Output: Three arrays of shape (N, 2, 19) or (N, 2, 4096)

all_head_scores  = np.zeros((N, 2, 19),   dtype=np.float32)
all_hidden_states = np.zeros((N, 2, 4096), dtype=np.float32)
all_gating       = np.zeros((N, 2, 19),   dtype=np.float32)

print(f"\nScoring {N} pairs ...")
for i, pair in enumerate(tqdm(pairs)):
    prompt = pair["prompt"]
    
    # Score chosen (axis 0)
    hs, hid, gate = score(prompt, pair["chosen_response"])
    all_head_scores[i, 0]   = hs
    all_hidden_states[i, 0] = hid
    all_gating[i, 0]        = gate
    
    # Score rejected (axis 1)
    hs, hid, gate = score(prompt, pair["rejected_response"])
    all_head_scores[i, 1]   = hs
    all_hidden_states[i, 1] = hid
    all_gating[i, 1]        = gate


# ── BLOCK 6: SANITY CHECK ────────────────────────────────────────────────────
# What: Print mean score gap (chosen - rejected) per head across ALL pairs.
# Why: If tokenization is correct, most heads should be positive (chosen > rejected).
#      If many heads are negative, something is wrong with input formatting.
#      This is the same check as in the hh-rlhf pipeline.

gaps = all_head_scores[:, 0, :] - all_head_scores[:, 1, :]  # (N, 19)
mean_gaps = gaps.mean(axis=0)

print("\nMean score gap (chosen - rejected) per head across all pairs:")
print(f"  {'Head':30s} {'Gap':>8s}  Direction")
print("  " + "-" * 50)
for name, gap in zip(HEAD_NAMES, mean_gaps):
    direction = "✓ chosen higher" if gap > 0 else "✗ rejected higher"
    print(f"  {name:30s} {gap:+.4f}  {direction}")

n_positive = (mean_gaps > 0).sum()
print(f"\n{n_positive}/19 heads score chosen > rejected on average.")


# ── BLOCK 7: SAVE OUTPUTS ────────────────────────────────────────────────────
# What: Save the three numpy arrays.
# Why: All downstream analysis (validity_check_helpsteer2.py) reads from these.
#      Same format as hh-rlhf and rewardbench outputs — analysis scripts
#      can be pointed at any of the three output directories.

os.makedirs(OUTPUT_DIR, exist_ok=True)

np.save(os.path.join(OUTPUT_DIR, "head_scores.npy"),    all_head_scores)
np.save(os.path.join(OUTPUT_DIR, "hidden_states.npy"),  all_hidden_states)
np.save(os.path.join(OUTPUT_DIR, "gating_outputs.npy"), all_gating)

print(f"\nSaved head_scores    → {OUTPUT_DIR}/head_scores.npy    shape: {all_head_scores.shape}")
print(f"Saved hidden_states  → {OUTPUT_DIR}/hidden_states.npy  shape: {all_hidden_states.shape}")
print(f"Saved gating_outputs → {OUTPUT_DIR}/gating_outputs.npy shape: {all_gating.shape}")
print("\nDone. Run validity_check_helpsteer2.py next.")
