# extract_scores_and_embeddings.py
#
# Entry point for Day 1-2.
# Run this script ONCE to score all pairs and save outputs/.
# Everything downstream reads from outputs/ — never re-runs the model.
#
# Usage:
#   python extract_scores_and_embeddings.py
#
# Expected runtime: ~45-55 min on Colab T4 (800 pairs x 2 passes each)
# After it finishes: zip and download outputs/ before session resets.
#
#   import shutil
#   shutil.make_archive("outputs_backup", "zip", "outputs")

import numpy as np
import os
from utils import (
    load_dataset_pairs,
    load_model_and_tokenizer,
    score_all_pairs,
    save_outputs,
    HEAD_NAMES,
    NUM_HEADS,
    SCORES_PATH,
    EMBEDDINGS_PATH,
    GATING_PATH,
    META_PATH,
)


def outputs_already_exist():
    return all(os.path.exists(p) for p in [SCORES_PATH, EMBEDDINGS_PATH, GATING_PATH, META_PATH])


def main():
    if outputs_already_exist():
        print("outputs/ already populated. Delete them to re-run extraction.")
        return

    # Step 1: Load dataset
    pairs = load_dataset_pairs()

    # Step 2: Load model
    tokenizer, model = load_model_and_tokenizer()

    # Step 3: Score all pairs
    print(f"\nScoring {len(pairs)} pairs across {NUM_HEADS} reward heads ...")
    print("This is the only time the model runs.\n")

    chosen_scores, rejected_scores, \
    chosen_hidden, rejected_hidden, \
    chosen_gating, rejected_gating = score_all_pairs(pairs, tokenizer, model)

    # Step 4: Sanity check
    print("\n── Sanity Check ──────────────────────────────────────────────────")
    print(f"chosen_scores shape  : {chosen_scores.shape}")    # (N, 19)
    print(f"rejected_scores shape: {rejected_scores.shape}")  # (N, 19)
    print(f"chosen_hidden shape  : {chosen_hidden.shape}")    # (N, 4096)
    print(f"chosen_gating shape  : {chosen_gating.shape}")    # (N, 19)

    score_gap = (chosen_scores - rejected_scores).mean(axis=0)
    print(f"\nMean score gap (chosen - rejected) per head:")
    print(f"  {'Head':<22}  {'Gap':>8}  {'Direction'}")
    print(f"  {'-'*45}")
    for name, gap in zip(HEAD_NAMES, score_gap):
        direction = "✓ chosen higher" if gap > 0 else "✗ rejected higher"
        print(f"  {name:<22}  {gap:>+8.4f}  {direction}")

    n_correct = (score_gap > 0).sum()
    print(f"\n{n_correct}/{NUM_HEADS} heads score chosen > rejected on average.")
    print("If this is below 10/19, check tokenize_text() in utils.py.\n")

    # Step 5: Save
    save_outputs(
        chosen_scores, rejected_scores,
        chosen_hidden, rejected_hidden,
        chosen_gating, rejected_gating,
        pairs
    )

    print("\nExtraction complete.")
    print("Next: run disentanglement_analysis.py, linear_separability.py,")
    print("      activation_analysis.py, validity_check.py")
    print("\nDon't forget to backup outputs/ before session resets:")
    print("  import shutil; shutil.make_archive('outputs_backup', 'zip', 'outputs')")


if __name__ == "__main__":
    main()