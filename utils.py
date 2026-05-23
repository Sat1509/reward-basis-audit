# utils.py
# Shared loading, scoring, and saving functions used across all scripts.
# Nothing runs from this file directly — it's imported by everything else.
#
# Key architectural detail:
#   ArmoRM has 19 reward heads, not the 8 mentioned in early HuggingFace docs.
#   outputs.rewards      -> (1, 19) raw head scores from regression_layer
#   outputs.hidden_state -> (1, 4096) pooled last-token embedding
#   outputs.gating_output-> (1, 19) context-dependent mixing weights
#   The final scalar = sum(gating * rewards @ transform_matrix).
#   We skip the final scalar and analyze the 19 heads directly.

import os
import json
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Constants:

MODEL_ID    = "RLHFlow/ArmoRM-Llama3-8B-v0.1"
DATASET_ID  = "Anthropic/hh-rlhf"
DATA_SPLIT  = "train"
MAX_SAMPLES = 800
MAX_LENGTH  = 1024
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

# ArmoRM's 19 reward head names in order matching the regression_layer output.
HEAD_NAMES = [
    "helpfulness",
    "correctness",
    "coherence",
    "complexity",
    "verbosity",
    "safety",
    "instruction_following",
    "honesty",
    "truthfulness",
    "harmlessness",
    "readability",
    "depth",
    "creativity",
    "detail",
    "positivity",
    "clarity",
    "engagement",
    "conciseness",
    "relevance",
]

NUM_HEADS = len(HEAD_NAMES)  # 19

# Output paths — all analysis scripts read from here after extraction.
SAVE_DIR        = "outputs"
SCORES_PATH     = os.path.join(SAVE_DIR, "head_scores.npy")
EMBEDDINGS_PATH = os.path.join(SAVE_DIR, "hidden_states.npy")
GATING_PATH     = os.path.join(SAVE_DIR, "gating_outputs.npy")
META_PATH       = os.path.join(SAVE_DIR, "metadata.json")

os.makedirs(SAVE_DIR, exist_ok=True)


# Dataset loading:
# We load MAX_SAMPLES chosen/rejected pairs from hh-rlhf.
# Each example is a dict with keys: 'chosen', 'rejected', 'index'.

def load_dataset_pairs(max_samples=MAX_SAMPLES):
    print(f"Loading dataset: {DATASET_ID} ...")
    dataset = load_dataset(DATASET_ID, split=DATA_SPLIT)

    pairs = []
    for i, example in enumerate(dataset):
        if i >= max_samples:
            break
        chosen   = example["chosen"].strip()
        rejected = example["rejected"].strip()
        if not chosen or not rejected:
            continue
        pairs.append({"chosen": chosen, "rejected": rejected, "index": i})

    print(f"Loaded {len(pairs)} pairs.")
    return pairs


# Model loading:
# We load in float16 to fit on a Colab T4 (15GB VRAM).
# trust_remote_code=True is required for ArmoRM's custom model class.

def load_model_and_tokenizer(model_id=MODEL_ID):
    print("Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True, use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model (this takes ~1-2 min) ...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="cuda"
    )
    model.eval()

    print(f"Model loaded on: {DEVICE}")
    print(f"Num reward heads: {NUM_HEADS}")
    return tokenizer, model


# Tokenization:
# hh-rlhf stores text as "Human: ...\n\nAssistant: ..."
# We split on the last Assistant turn and apply ArmoRM's chat template.

# Note: ArmoRM's gating network calls find_token_for_gating() internally,
# which looks for the token pattern [128009, 128006, 78191, 128007, 271].
# apply_chat_template guarantees this pattern is present. Raw tokenization breaks it.

def tokenize_text(text, tokenizer, max_length=MAX_LENGTH):
    if "\n\nAssistant:" in text:
        prompt_part, response_part = text.rsplit("\n\nAssistant:", 1)
        prompt_part   = prompt_part.replace("Human:", "").strip()
        response_part = response_part.strip()
    else:
        prompt_part   = text.strip()
        response_part = ""

    messages = [
        {"role": "user",      "content": prompt_part},
        {"role": "assistant", "content": response_part}
    ]

    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )

    encoded = tokenizer(
        formatted,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=False
    )

    return {k: v.to(DEVICE) for k, v in encoded.items()}


#Single-example scoring:
# One forward pass through ArmoRM. Returns all three outputs we need:
#   head_scores  (19,)   — raw reward head scores
#   hidden_state (4096,) — pooled last-token embedding
#   gating       (19,)   — softmax mixing weights from GatingNetwork

def score_text(text, tokenizer, model):
    inputs = tokenize_text(text, tokenizer)

    with torch.no_grad():
        outputs = model(**inputs)

    head_scores  = outputs.rewards[0].cpu().float().numpy()
    hidden_state = outputs.hidden_state[0].cpu().float().numpy()
    gating       = outputs.gating_output[0].cpu().float().numpy()

    return head_scores, hidden_state, gating


# Batch scoring:
# Runs score_text on every chosen and rejected response.
# Returns six arrays, all row-aligned to pairs[i].
# Runtime: ~45-55 min for 800 pairs on a T4.

def score_all_pairs(pairs, tokenizer, model):
    chosen_scores,  rejected_scores  = [], []
    chosen_hidden,  rejected_hidden  = [], []
    chosen_gating,  rejected_gating  = [], []

    total = len(pairs)
    for i, pair in enumerate(pairs):
        if i % 50 == 0:
            print(f"  Scoring pair {i}/{total} ...")

        c_scores, c_hidden, c_gating = score_text(pair["chosen"],   tokenizer, model)
        r_scores, r_hidden, r_gating = score_text(pair["rejected"], tokenizer, model)

        chosen_scores.append(c_scores);  rejected_scores.append(r_scores)
        chosen_hidden.append(c_hidden);  rejected_hidden.append(r_hidden)
        chosen_gating.append(c_gating);  rejected_gating.append(r_gating)

    return (
        np.array(chosen_scores),    # (N, 19)
        np.array(rejected_scores),
        np.array(chosen_hidden),    # (N, 4096)
        np.array(rejected_hidden),
        np.array(chosen_gating),    # (N, 19)
        np.array(rejected_gating)
    )


# Save outputs:
# Arrays are stacked as (N, 2, dim) — axis 1 is [chosen=0, rejected=1].
# This convention is used consistently across all analysis scripts.

def save_outputs(chosen_scores, rejected_scores,
                 chosen_hidden, rejected_hidden,
                 chosen_gating, rejected_gating,
                 pairs):
    scores  = np.stack([chosen_scores, rejected_scores], axis=1)
    hiddens = np.stack([chosen_hidden, rejected_hidden], axis=1)
    gatings = np.stack([chosen_gating, rejected_gating], axis=1)

    np.save(SCORES_PATH,     scores)
    np.save(EMBEDDINGS_PATH, hiddens)
    np.save(GATING_PATH,     gatings)

    meta = {
        "head_names": HEAD_NAMES,
        "num_heads":  NUM_HEADS,
        "num_pairs":  len(pairs),
        "model_id":   MODEL_ID,
        "dataset_id": DATASET_ID,
        "pairs": [
            {"index": p["index"], "chosen": p["chosen"][:200], "rejected": p["rejected"][:200]}
            for p in pairs
        ]
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved scores       -> {SCORES_PATH}    shape: {scores.shape}")
    print(f"Saved hidden states-> {EMBEDDINGS_PATH} shape: {hiddens.shape}")
    print(f"Saved gating       -> {GATING_PATH}     shape: {gatings.shape}")
    print(f"Saved metadata     -> {META_PATH}")


# Load outputs :
# All analysis scripts call this instead of re-running the model.
# Note: if SAVE_DIR is overridden in a script (e.g. for rewardbench outputs),
# update SCORES_PATH etc. before calling this function.

def load_outputs():
    scores  = np.load(SCORES_PATH)
    hiddens = np.load(EMBEDDINGS_PATH)
    gatings = np.load(GATING_PATH)

    with open(META_PATH, "r") as f:
        meta = json.load(f)

    head_names = meta.get("head_names", HEAD_NAMES)
    print(f"Loaded {scores.shape[0]} pairs | {scores.shape[2]} heads")
    print(f"Heads: {head_names}")

    return (
        scores[:,  0, :],   # chosen_scores   (N, 19)
        scores[:,  1, :],   # rejected_scores (N, 19)
        hiddens[:, 0, :],   # chosen_hidden   (N, 4096)
        hiddens[:, 1, :],   # rejected_hidden (N, 4096)
        gatings[:, 0, :],   # chosen_gating   (N, 19)
        gatings[:, 1, :],   # rejected_gating (N, 19)
        meta
    )
