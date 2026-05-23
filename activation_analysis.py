# activation_analysis.py
#
# Research question answered here:
#   Which tokens and layers drive each reward head?
#   Does the "safety" head actually activate more on safety-relevant tokens
#   (refusals, warnings, harmful content markers)?
#   Does "helpfulness" activate on substantive answer tokens?
#
#   This is the mechanistic interpretability core — we're not just asking
#   "does the head score high?" but "what in the input causes it to score high?"
#
# Method:
#   Input x Output Gradient attribution — for each head, we compute the
#   gradient of that head's score with respect to each token's embedding.
#   Tokens with high gradient magnitude = tokens that most influence that head.
#
#   This is the same family as Integrated Gradients (which you know from your
#   XAI background), but simpler: one-shot gradient rather than path integral.
#   Mention this connection explicitly to Rachel — it shows methodological continuity.
#
# Outputs:
#   outputs/activation_top_tokens.json     — top influential tokens per head
#   outputs/activation_layer_norms.npy     — per-layer activation norms
#   outputs/activation_examples.txt        — human-readable token attribution examples
#
# Usage:
#   python activation_analysis.py
#   (Requires outputs/ populated by extract_scores_and_embeddings.py)
#   NOTE: This script re-loads the model because it needs gradients.
#         Run on GPU. Expects ~20-30 min for 200 examples.

import numpy as np
import json
import os
import torch
import collections
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from utils import (
    load_dataset_pairs, load_model_and_tokenizer,
    tokenize_text, HEAD_NAMES, SAVE_DIR,
    MAX_SAMPLES, MODEL_ID
)

TOP_TOKENS_PATH   = os.path.join(SAVE_DIR, "activation_top_tokens.json")
LAYER_NORMS_PATH  = os.path.join(SAVE_DIR, "activation_layer_norms.npy")
EXAMPLES_PATH     = os.path.join(SAVE_DIR, "activation_examples.txt")

# We run attribution on a subset — full 800 is too slow with gradients
ATTRIBUTION_SUBSET = 200

# Top N tokens to track per head
TOP_N = 20

# Safety-relevant keywords we'll check against top tokens
# If the "safety" head's top tokens overlap with these, the label is valid
SAFETY_KEYWORDS = {
    "sorry", "cannot", "can't", "unsafe", "harmful", "illegal",
    "dangerous", "refuse", "inappropriate", "not", "don't", "warn",
    "caution", "risk", "avoid", "never", "wrong", "unethical"
}

HELPFULNESS_KEYWORDS = {
    "here", "sure", "happy", "help", "answer", "explain", "provide",
    "steps", "first", "second", "solution", "example", "following"
}

HEAD_KEYWORDS = {
    "safety":              SAFETY_KEYWORDS,
    "helpfulness":         HELPFULNESS_KEYWORDS,
    "honesty":             {"actually", "fact", "true", "false", "incorrect", "accurate", "note", "however"},
    "instruction_following": {"as", "requested", "asked", "per", "following", "instructions", "format"},
}


# ── Block 1: Gradient attribution for one example ─────────────────────────────

def attribute_tokens(text, tokenizer, model, head_idx):
    """
    For a single text and a single reward head:
    1. Tokenize the text
    2. Forward pass with gradient tracking enabled
    3. Compute gradient of head_idx score w.r.t. each token embedding
    4. Attribution score = L2 norm of gradient vector at each token position

    Returns:
        tokens      : list of decoded token strings
        attributions: np.array of shape (seq_len,) — one score per token
    """
    inputs = tokenize_text(text, tokenizer)

    # We need gradients w.r.t. embeddings, not weights
    # Get the embedding layer output and enable grad
    model.eval()

    # Hook to capture embeddings
    embeddings_ref = {}

    def embedding_hook(module, input, output):
        embeddings_ref["embeddings"] = output
        output.retain_grad()

    # Register hook on the embedding layer
    # For Llama3 this is model.model.embed_tokens
    hook = model.model.embed_tokens.register_forward_hook(embedding_hook)

    try:
        outputs = model(**inputs)
        head_score = outputs.logits[0, head_idx]  # Scalar score for this head

        # Backpropagate through the head score
        model.zero_grad()
        head_score.backward()

        # Gradient at each token position: shape (seq_len, hidden_size)
        grads = embeddings_ref["embeddings"].grad  # (1, seq_len, hidden_size)

        if grads is None:
            return None, None

        grads = grads[0].detach().cpu().float().numpy()  # (seq_len, hidden_size)

        # Attribution = L2 norm of gradient at each token
        attributions = np.linalg.norm(grads, axis=-1)  # (seq_len,)

        # Decode tokens
        input_ids = inputs["input_ids"][0].cpu().numpy()
        tokens = [tokenizer.decode([tid]) for tid in input_ids]

    finally:
        hook.remove()

    return tokens, attributions


# ── Block 2: Aggregate top tokens across examples ─────────────────────────────

def aggregate_top_tokens(pairs, tokenizer, model, subset=ATTRIBUTION_SUBSET):
    """
    Runs token attribution on `subset` chosen responses for all 8 heads.
    For each head, accumulates a frequency-weighted token importance score:
        token_importance[head][token] += attribution_score

    After aggregating, returns top-N tokens per head by total importance.

    We use chosen responses only — they're the positive signal we want
    to understand. Rejected responses are the negative contrastive signal
    handled in validity_check.py.
    """
    # token_scores[head_name][token_string] = cumulative attribution
    token_scores = {h: collections.defaultdict(float) for h in HEAD_NAMES}

    subset_pairs = pairs[:subset]
    total = len(subset_pairs)

    for i, pair in enumerate(subset_pairs):
        if i % 20 == 0:
            print(f"  Attributing example {i}/{total} ...")

        text = pair["chosen"]

        for head_idx, head_name in enumerate(HEAD_NAMES):
            tokens, attributions = attribute_tokens(text, tokenizer, model, head_idx)

            if tokens is None:
                continue

            # Normalize attributions to sum to 1 for this example
            total_attr = attributions.sum()
            if total_attr == 0:
                continue
            attributions = attributions / total_attr

            for token, score in zip(tokens, attributions):
                # Clean token: strip whitespace, lowercase
                clean = token.strip().lower()
                if len(clean) < 2:   # Skip single chars and punctuation
                    continue
                token_scores[head_name][clean] += float(score)

    # Get top-N per head
    top_tokens = {}
    for head_name in HEAD_NAMES:
        sorted_tokens = sorted(
            token_scores[head_name].items(),
            key=lambda x: x[1],
            reverse=True
        )[:TOP_N]
        top_tokens[head_name] = [
            {"token": t, "score": round(s, 4)}
            for t, s in sorted_tokens
        ]

    return top_tokens


# ── Block 3: Keyword overlap analysis ─────────────────────────────────────────

def compute_keyword_overlap(top_tokens):
    """
    For heads where we have semantic expectations (safety, helpfulness, etc.),
    check what fraction of the top-N tokens are semantically relevant keywords.

    High overlap = the head activates on semantically appropriate tokens.
    Low overlap  = the head activates on arbitrary / syntactic tokens.
                   The label does not correspond to what drives the head.

    This is the token-level validity check — complements validity_check.py
    which operates at the response level.
    """
    overlap_results = {}

    for head_name, keywords in HEAD_KEYWORDS.items():
        if head_name not in top_tokens:
            continue

        head_top = {entry["token"] for entry in top_tokens[head_name]}
        overlap  = head_top & keywords
        overlap_frac = len(overlap) / TOP_N

        overlap_results[head_name] = {
            "top_tokens":     [e["token"] for e in top_tokens[head_name]],
            "expected_keywords": list(keywords),
            "matching_tokens":   list(overlap),
            "overlap_fraction":  round(overlap_frac, 3),
            "interpretation": (
                "label-consistent" if overlap_frac >= 0.2
                else "label-inconsistent — head drives on unexpected tokens"
            )
        }

    return overlap_results


# ── Block 4: Save outputs ─────────────────────────────────────────────────────

def save_outputs(top_tokens, overlap_results):
    """
    Saves top tokens per head and keyword overlap analysis to JSON.
    Also writes a human-readable text file for quick inspection.
    """
    output = {
        "analysis":        "activation_analysis",
        "method":          "Input x gradient attribution, L2 norm per token",
        "attribution_subset": ATTRIBUTION_SUBSET,
        "top_n_tokens":    TOP_N,
        "top_tokens_per_head": top_tokens,
        "keyword_overlap": overlap_results
    }

    with open(TOP_TOKENS_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Top tokens saved -> {TOP_TOKENS_PATH}")

    # Human-readable summary
    with open(EXAMPLES_PATH, "w") as f:
        f.write("ArmoRM Activation Analysis — Top Tokens Per Reward Head\n")
        f.write("=" * 60 + "\n\n")

        for head_name in HEAD_NAMES:
            tokens = top_tokens.get(head_name, [])
            f.write(f"[{head_name.upper()}]\n")
            f.write("  Top tokens: " + ", ".join(e["token"] for e in tokens[:10]) + "\n")

            if head_name in overlap_results:
                ov = overlap_results[head_name]
                f.write(f"  Keyword overlap: {ov['overlap_fraction']:.0%} "
                        f"({ov['interpretation']})\n")
                if ov["matching_tokens"]:
                    f.write(f"  Matching: {', '.join(ov['matching_tokens'])}\n")
            f.write("\n")

    print(f"Readable summary -> {EXAMPLES_PATH}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading dataset ...")
    pairs = load_dataset_pairs()

    print("\nLoading model (gradients required — float16 + no_grad won't work here) ...")
    # For gradient attribution we need float32 on at least the embedding layer
    # We reload the model here without float16 quantization for correctness
    # This uses more VRAM — if OOM, reduce ATTRIBUTION_SUBSET to 100
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype=torch.float32,   # float32 for gradient stability
        device_map="auto"
    )
    # Do NOT call model.eval() fully — we need gradients
    # But disable dropout manually
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0

    print(f"\nRunning token attribution on {ATTRIBUTION_SUBSET} examples ...")
    print("(~20-30 min on T4 — reduce ATTRIBUTION_SUBSET to 100 if slow)\n")

    top_tokens = aggregate_top_tokens(pairs, tokenizer, model, ATTRIBUTION_SUBSET)

    print("\nComputing keyword overlap for semantic validity ...")
    overlap_results = compute_keyword_overlap(top_tokens)

    print("\nSaving outputs ...")
    save_outputs(top_tokens, overlap_results)

    print("\n── Activation Analysis Summary ───────────────────────────────────")
    for head_name in HEAD_NAMES:
        tokens_str = ", ".join(e["token"] for e in top_tokens.get(head_name, [])[:8])
        print(f"  {head_name:<22} top tokens: {tokens_str}")

    print("\n── Keyword Overlap (label validity at token level) ───────────────")
    for head_name, result in overlap_results.items():
        print(f"  {head_name:<22} {result['overlap_fraction']:.0%} overlap  |  {result['interpretation']}")

    print("\nHow to read this for Rachel:")
    print("  If 'safety' head's top tokens are 'cannot', 'sorry', 'harmful' -> label is valid.")
    print("  If 'safety' head's top tokens are 'the', 'and', 'user' -> the head drives on")
    print("  syntactic/positional features, not semantic safety content.")
    print("  That is a mechanistic interpretability finding: the label is not grounded.")


if __name__ == "__main__":
    main()
