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


# ---

print("Loading metadata ...")
with open(os.path.join(INPUT_DIR, "metadata.json")) as f:
    meta = json.load(f)

pairs = meta["pairs"]
N = len(pairs)
print(f"  {N} pairs to score")


# ---
# float16, same as hh-rlhf pipeline — identical model keeps the dataset comparison fair

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


# ---
# gating network requires chat template — same reason as hh-rlhf pipeline (token pattern lookup)

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


# ---
# head_scores for validity; hidden_states for probing; gating_outputs to inspect per-head weighting

@torch.no_grad()
def score(prompt, response):
    tokens = tokenize(prompt, response)
    outputs = model(**tokens)
    head_scores = outputs.rewards[0].cpu().float().numpy()       # (19,)
    hidden_state = outputs.hidden_states[0].cpu().float().numpy() # (4096,)
    gating = outputs.gating_output[0].cpu().float().numpy()       # (19,)
    return head_scores, hidden_state, gating


# ---
# axis 1: [chosen=0, rejected=1]

all_head_scores  = np.zeros((N, 2, 19),   dtype=np.float32)
all_hidden_states = np.zeros((N, 2, 4096), dtype=np.float32)
all_gating       = np.zeros((N, 2, 19),   dtype=np.float32)

print(f"\nScoring {N} pairs ...")
for i, pair in enumerate(tqdm(pairs)):
    prompt = pair["prompt"]
    
    hs, hid, gate = score(prompt, pair["chosen_response"])   # axis 0: chosen
    all_head_scores[i, 0]   = hs
    all_hidden_states[i, 0] = hid
    all_gating[i, 0]        = gate

    hs, hid, gate = score(prompt, pair["rejected_response"])  # axis 1: rejected
    all_head_scores[i, 1]   = hs
    all_hidden_states[i, 1] = hid
    all_gating[i, 1]        = gate


# ---
# if tokenization is correct, most heads should show chosen > rejected

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


# ---
# same format as hh-rlhf and rewardbench outputs — analysis scripts are dataset-agnostic

os.makedirs(OUTPUT_DIR, exist_ok=True)

np.save(os.path.join(OUTPUT_DIR, "head_scores.npy"),    all_head_scores)
np.save(os.path.join(OUTPUT_DIR, "hidden_states.npy"),  all_hidden_states)
np.save(os.path.join(OUTPUT_DIR, "gating_outputs.npy"), all_gating)

print(f"\nSaved head_scores    → {OUTPUT_DIR}/head_scores.npy    shape: {all_head_scores.shape}")
print(f"Saved hidden_states  → {OUTPUT_DIR}/hidden_states.npy  shape: {all_hidden_states.shape}")
print(f"Saved gating_outputs → {OUTPUT_DIR}/gating_outputs.npy shape: {all_gating.shape}")
print("\nDone. Run validity_check_helpsteer2.py next.")
