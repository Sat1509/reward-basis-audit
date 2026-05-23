"""
geometry_analysis.py

Research question:
    ArmoRM's 19 reward heads are behaviorally correlated (shown in disentanglement_analysis.py).
    But is that entanglement architectural — baked into the weight matrix itself —
    or data-driven — a property of the training distribution that the architecture
    could in principle separate?

    This distinction has direct implications for what "fixing" the decomposition requires.
    Data-driven entanglement is fixable with targeted contrastive data or fine-tuning.
    Architecture-driven entanglement requires redesigning the reward head structure.

Approach:
    The regression_layer is Linear(4096 → 19). Its weight matrix W ∈ ℝ^{19×4096}
    defines 19 directions in hidden space — one per reward head. We audit the
    geometric structure of those directions directly, then compare to behavioral
    correlations from saved head scores.

Five analyses:
    1. Gram matrix + cosine similarity  → are weight vectors orthogonal?
    2. SVD + effective rank             → how many independent directions do the 19 heads span?
    3. Principal angles                 → does the scoring subspace align with the
                                          representation subspace (PCA of hidden states)?
    4. Entanglement decomposition       → what fraction of behavioral correlation is predicted
                                          by geometric overlap? This separates architecture-driven
                                          from data-driven entanglement.
    5. Projection variance              → how much of hidden state variance lives in the
                                          weight-defined scoring subspace?

All analyses use:
    W  ∈ ℝ^{19×4096}      — regression layer weight matrix (extracted from model once, cached)
    H  ∈ ℝ^{1600×4096}    — hidden states, 800 pairs × 2 responses, from outputs/
    Scores ∈ ℝ^{1600×19}  — head scores, from outputs/

No GPU required after W is cached. Everything here is CPU linear algebra.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from sklearn.decomposition import PCA
import json
import os
import sys

# ── Constants ──────────────────────────────────────────────────────────────────

HEAD_NAMES = [
    'helpfulness', 'correctness', 'coherence', 'complexity', 'verbosity',
    'safety', 'instruction_following', 'honesty', 'truthfulness', 'harmlessness',
    'readability', 'depth', 'creativity', 'detail', 'positivity', 'clarity',
    'engagement', 'conciseness', 'relevance'
]
N = len(HEAD_NAMES)   # 19

os.makedirs('outputs', exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# BLOCK 1: Load regression layer weights
# ══════════════════════════════════════════════════════════════════════════════
#
# The regression_layer is a Linear(4096 → 19) layer with no bias.
# Its weight matrix W has shape (19, 4096).
# Row i of W is the weight vector w_i for head i — a direction in the 4096-dim
# hidden space. The raw score for head i is:
#
#   score_i(h) = w_i · h     (dot product with the pooled hidden state h)
#
# This is the object we analyze geometrically. Everything downstream is
# operations on W and on the saved hidden states H.
#
# We cache W to outputs/regression_weights.npy so the model doesn't need
# to be reloaded. Alternatively, add one line to extract_scores_and_embeddings.py:
#   np.save('outputs/regression_weights.npy',
#           model.regression_layer.weight.detach().float().cpu().numpy())

def load_regression_weights():
    """
    Returns W ∈ ℝ^{19×4096}.
    Loads from cache if available; otherwise loads full model, extracts W, saves cache.
    """
    cache = 'outputs/regression_weights.npy'
    if os.path.exists(cache):
        W = np.load(cache)
        print(f"Loaded cached regression weights    shape: {W.shape}")
        return W

    print("Cache not found — loading model to extract regression_layer.weight ...")
    try:
        # Reuse utils.py if available — avoids duplicating model loading logic
        sys.path.insert(0, '.')
        from utils import load_model_and_tokenizer
        model, _ = load_model_and_tokenizer()
        W = model.regression_layer.weight.detach().float().cpu().numpy()
    except ImportError:
        # Fallback: standalone load. CPU only — we just need the weight matrix.
        import torch
        from transformers import AutoModelForSequenceClassification
        model = AutoModelForSequenceClassification.from_pretrained(
            "RLHFlow/ArmoRM-Llama3-8B-v0.1",
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="cpu",
        )
        W = model.regression_layer.weight.detach().float().numpy()
    
    np.save(cache, W)
    print(f"Saved regression weights to cache   shape: {W.shape}")
    del model
    return W


# ══════════════════════════════════════════════════════════════════════════════
# BLOCK 2: Gram matrix and cosine similarity
# ══════════════════════════════════════════════════════════════════════════════
#
# The Gram matrix G = W Wᵀ has entries:
#   G_ij = wᵢ · wⱼ   (raw dot product between head i and head j weight vectors)
#
# Normalizing each row of W to unit length gives Ŵ (unit-norm rows).
# The cosine similarity matrix is:
#   C = Ŵ Ŵᵀ   where C_ij = (wᵢ · wⱼ) / (‖wᵢ‖ ‖wⱼ‖) ∈ [-1, 1]
#   C_ii = 1 by construction.
#
# Interpretation:
#   C ≈ I  →  the 19 weight vectors are mutually orthogonal. The architecture
#              defines 19 independent directions. Any behavioral correlation
#              between heads must be a property of the data, not the architecture.
#
#   Large off-diagonal C_ij  →  weight vectors share directions. The model
#              collapsed these concepts at the architecture level.
#
# Orthogonality gap: ‖C − I‖_F / √(N(N−1))
#   Normalizing by √(N(N−1)) accounts for the N(N−1) off-diagonal entries,
#   giving a value in [0, 1] where 0 = perfect orthogonality.

def gram_analysis(W):
    """
    Args:
        W : (N, 4096) weight matrix

    Returns:
        C        : (N, N) cosine similarity matrix of weight vectors
        G        : (N, N) raw Gram matrix
        orth_gap : scalar ∈ [0,1], 0 = perfectly orthogonal
        norms    : (N,) L2 norm of each row of W
    """
    norms = np.linalg.norm(W, axis=1)              # (N,)
    W_hat = W / norms[:, np.newaxis]               # (N, 4096) unit-norm rows

    G = W @ W.T                                    # (N, N) raw Gram matrix
    C = W_hat @ W_hat.T                            # (N, N) cosine similarity

    offdiag = C - np.eye(N)
    orth_gap = np.linalg.norm(offdiag, 'fro') / np.sqrt(N * (N - 1))

    return C, G, orth_gap, norms


# ══════════════════════════════════════════════════════════════════════════════
# BLOCK 3: SVD and effective rank
# ══════════════════════════════════════════════════════════════════════════════
#
# The singular value decomposition of W is:
#   W = U Σ Vᵀ
#   U ∈ ℝ^{N×N}      — left singular vectors (how heads mix into components)
#   Σ ∈ ℝ^{N×N}      — diagonal, singular values σ₁ ≥ σ₂ ≥ ... ≥ σ_N ≥ 0
#   Vᵀ ∈ ℝ^{N×4096}  — right singular vectors (actual directions in hidden space)
#
# Each row of Vᵀ is one independent direction the model uses for scoring.
# Each σᵢ is the magnitude of that direction — how much "signal" is in it.
#
# If all σᵢ are equal → W has full rank N, all 19 directions are equally used.
# If σᵢ drops to near zero for i > k → the 19 heads effectively span only a
# k-dimensional subspace. They are not 19 independent axes.
#
# Effective rank measures (Roy & Vetterli, 2007):
#
#   Participation ratio = (Σ σᵢ)² / Σ σᵢ²
#     = N if all singular values equal (full effective rank)
#     = 1 if one singular value dominates (rank-1 effective structure)
#
#   Entropy rank = exp(−Σ pᵢ log pᵢ)  where pᵢ = σᵢ² / Σ σⱼ²
#     = Shannon entropy of the squared singular value distribution
#     = N for uniform spectrum (maximum diversity), = 1 for peaked spectrum
#
#   Condition number κ = σ_max / σ_min
#     κ >> 1 → near-linear-dependence among rows → ill-conditioned weight matrix
#     Geometrically: some heads are almost linear combinations of others.

def svd_analysis(W):
    """
    Args:
        W : (N, 4096) weight matrix

    Returns:
        U, S, Vt : SVD components. S shape (N,).
        metrics  : dict of effective rank statistics
    """
    U, S, Vt = np.linalg.svd(W, full_matrices=False)    # S: (N,)

    # Squared singular values (proportional to variance explained)
    S2 = S ** 2
    p = S2 / S2.sum()                                    # probability distribution over components

    participation_ratio = S.sum() ** 2 / S2.sum()
    entropy_rank = float(np.exp(-np.sum(p * np.log(p + 1e-12))))
    condition_number = float(S[0] / S[-1])

    explained = S2 / S2.sum()
    cumulative = np.cumsum(explained)
    k90 = int(np.searchsorted(cumulative, 0.90)) + 1     # components needed for 90% of W's variance

    metrics = {
        'singular_values'      : S.tolist(),
        'explained_variance'   : explained.tolist(),
        'cumulative_variance'  : cumulative.tolist(),
        'participation_ratio'  : float(participation_ratio),
        'entropy_rank'         : entropy_rank,
        'condition_number'     : condition_number,
        'k90'                  : k90,
    }
    return U, S, Vt, metrics


# ══════════════════════════════════════════════════════════════════════════════
# BLOCK 4: Principal angles between subspaces
# ══════════════════════════════════════════════════════════════════════════════
#
# The principal angles between two subspaces A and B in ℝ^d generalize the
# concept of angle between two vectors to angle between two subspaces.
#
# We compare:
#   A: row space of W      (directions the model uses for scoring)
#   B: top-k PCA subspace  (directions of maximum variance in hidden states)
#
# If the scoring directions align with the high-variance representation directions
# (small principal angles), the model scores along features that dominate its
# internal geometry — a coherent picture.
#
# If A ⊥ B (large principal angles), the model scores along directions nearly
# orthogonal to the hidden state variance — a dissonant picture: high probe
# accuracy is only possible because some of the 20% un-explained variance lives
# in the scoring subspace.
#
# Algorithm (Björck & Golub, 1973):
#   1. Orthonormalize rows of A via QR → Q_A ∈ ℝ^{d × r_A}
#   2. Orthonormalize rows of B via QR → Q_B ∈ ℝ^{d × r_B}
#   3. Compute M = Q_Aᵀ Q_B ∈ ℝ^{r_A × r_B}  (inner products between bases)
#   4. SVD of M → singular values σᵢ = cos(θᵢ)
#   θᵢ = arccos(σᵢ) ∈ [0°, 90°],  θ = 0° means the subspaces share a direction.

def principal_angles(A_rows, B_rows):
    """
    Compute principal angles between subspace spanned by rows of A_rows
    and subspace spanned by rows of B_rows.

    Args:
        A_rows : (m, d) — m vectors in ℝ^d spanning subspace A
        B_rows : (n, d) — n vectors in ℝ^d spanning subspace B

    Returns:
        angles_deg : (k,) principal angles in degrees, k = min(rank A, rank B)
    """
    # QR on transposed matrices: columns of A_rows.T are the spanning vectors
    Q_A, _ = np.linalg.qr(A_rows.T)     # (d, rank_A), orthonormal basis for row space of A_rows
    Q_B, _ = np.linalg.qr(B_rows.T)     # (d, rank_B)

    M = Q_A.T @ Q_B                      # (rank_A, rank_B) inner product matrix
    sigma = np.linalg.svd(M, compute_uv=False)
    sigma = np.clip(sigma, 0.0, 1.0)     # numerical safety before arccos

    return np.degrees(np.arccos(sigma))


# ══════════════════════════════════════════════════════════════════════════════
# BLOCK 5: Entanglement decomposition
# ══════════════════════════════════════════════════════════════════════════════
#
# The core comparison of this script.
#
# We have two matrices, both (N, N):
#   C_weight[i,j] = cosine similarity of weight vectors wᵢ and wⱼ
#                   (geometric overlap in weight space — architecture property)
#   C_score[i,j]  = Spearman rank correlation of head i scores vs head j scores
#                   (behavioral correlation — joint property of architecture + data)
#
# We regress C_score on C_weight using the N(N−1)/2 = 171 upper-triangle pairs.
#
# The R² of this regression answers: what fraction of behavioral entanglement
# variance is explained by geometric overlap in the weight matrix?
#
#   R² ≈ 0 → entanglement is DATA-DRIVEN.
#     The weight vectors are orthogonal. The architecture CAN represent independent
#     reward concepts. But the training distribution never provided inputs where
#     these concepts diverge, so the heads always fire together.
#     Fix: targeted contrastive data (e.g., responses that are safe but not honest,
#     or helpful but not coherent). This is a data problem, not a model problem.
#
#   R² ≈ 1 → entanglement is ARCHITECTURE-DRIVEN.
#     The weight vectors themselves share directions. The model has geometrically
#     conflated these concepts. Even with perfect training data, the heads would
#     be correlated because they point in similar directions.
#     Fix: orthogonality regularization on W during training, or reparametrizing
#     the reward heads to enforce orthogonal directions.
#
#   Intermediate R² → both sources contribute.

def entanglement_decomposition(W, scores_flat):
    """
    Args:
        W           : (N, 4096) weight matrix
        scores_flat : (M, N) head scores pooled across all responses

    Returns:
        C_weight    : (N, N) cosine similarity matrix of weight vectors
        C_score     : (N, N) Spearman rank correlation matrix of head scores
        slope, intercept, r2 : linear regression of C_score on C_weight (upper triangle)
        cos_tri, sp_tri : (171,) upper-triangle values for scatter plot
    """
    # Weight-space geometry
    norms = np.linalg.norm(W, axis=1)
    W_hat = W / norms[:, np.newaxis]
    C_weight = W_hat @ W_hat.T              # (N, N)

    # Score-space correlation
    C_score = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            C_score[i, j], _ = stats.spearmanr(scores_flat[:, i], scores_flat[:, j])

    # Upper triangle (excluding diagonal) — 171 pairs
    idx = np.triu_indices(N, k=1)
    cos_tri = C_weight[idx]
    sp_tri  = C_score[idx]

    slope, intercept, r, p_val, _ = stats.linregress(cos_tri, sp_tri)
    r2 = r ** 2

    return C_weight, C_score, float(slope), float(intercept), float(r2), cos_tri, sp_tri


# ══════════════════════════════════════════════════════════════════════════════
# BLOCK 6: Projection variance
# ══════════════════════════════════════════════════════════════════════════════
#
# The rows of W span a subspace S_W ⊆ ℝ^{4096} of dimension rank(W) ≤ 19.
# We ask: what fraction of the variance in the hidden states H lives within S_W?
#
# Procedure:
#   1. Get an orthonormal basis Q for S_W via SVD of W:
#      Right singular vectors Vᵀ → rows are directions in ℝ^{4096}
#      Keep those with σᵢ > threshold → Q ∈ ℝ^{4096 × r}
#   2. Project H onto S_W:  H_proj = H Q Qᵀ
#   3. Projection variance = ‖H_proj‖_F² / ‖H_centered‖_F²
#
# Compare to: variance explained by the top-r PCA components of H.
# PCA gives the r-dimensional subspace that maximizes explained variance.
# S_W is the r-dimensional subspace the model actually uses for scoring.
#
# Gap = var_pca_r − var_scoring_r ≥ 0
#   Gap ≈ 0 → the scoring subspace is as informative as the best possible
#              r-dimensional subspace. The model scores along its most variable directions.
#   Gap >> 0 → the model uses directions that aren't the most variable in H.
#              Scoring signal lives in a quieter corner of representation space.
#
# Note: this is a property of alignment between W and H, not of W alone.

def projection_variance(W, H):
    """
    Args:
        W : (N, 4096) weight matrix
        H : (M, 4096) hidden states (will be centered internally)

    Returns:
        var_scoring : fraction of H variance in W's row space
        var_pca_r   : fraction of H variance in top-r PCA subspace (r = rank(W))
        rank_W      : effective rank of W (number of singular values > 1e-6)
    """
    H_c = H - H.mean(axis=0, keepdims=True)          # center hidden states

    # Orthonormal basis for W's row space from SVD
    _, S_w, Vt_w = np.linalg.svd(W, full_matrices=False)
    rank_W = int(np.sum(S_w > 1e-6))
    Q = Vt_w[:rank_W].T                              # (4096, rank_W)

    # Project H onto S_W
    H_proj = H_c @ Q @ Q.T                           # (M, 4096)

    total_var   = np.linalg.norm(H_c, 'fro') ** 2
    proj_var    = np.linalg.norm(H_proj, 'fro') ** 2
    var_scoring = proj_var / total_var

    # PCA comparison: best possible r-dimensional subspace
    pca = PCA(n_components=rank_W)
    pca.fit(H_c)
    var_pca_r = float(pca.explained_variance_ratio_.sum())

    return float(var_scoring), var_pca_r, rank_W


# ══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

def plot_all(C_weight, S, svd_metrics, angles, cos_tri, sp_tri, slope, intercept, r2):
    fig = plt.figure(figsize=(18, 14))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35)

    # ── Panel 1: Weight-space cosine similarity heatmap ────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    im = ax1.imshow(C_weight, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax1.set_xticks(range(N)); ax1.set_yticks(range(N))
    ax1.set_xticklabels(HEAD_NAMES, rotation=90, fontsize=7)
    ax1.set_yticklabels(HEAD_NAMES, fontsize=7)
    ax1.set_title('Weight-Space Cosine Similarity\nC = ŴŴᵀ,  C_ij = cos(wᵢ, wⱼ)',
                  fontsize=10, fontweight='bold')
    plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

    # ── Panel 2: Singular value spectrum of W ──────────────────────────────
    ax2  = fig.add_subplot(gs[0, 1])
    exp  = np.array(svd_metrics['explained_variance']) * 100
    cum  = np.array(svd_metrics['cumulative_variance']) * 100
    ax2.bar(range(1, N+1), exp, color='steelblue', alpha=0.75, label='Per component')
    ax2r = ax2.twinx()
    ax2r.plot(range(1, N+1), cum, 'r-o', markersize=4, label='Cumulative')
    ax2r.axhline(90, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax2r.set_ylabel('Cumulative variance (%)', color='red', fontsize=9)
    ax2r.tick_params(axis='y', labelcolor='red')
    ax2.set_xlabel('Singular value index', fontsize=9)
    ax2.set_ylabel('Variance explained (%)', fontsize=9)
    ax2.set_xticks(range(1, N+1))
    ax2.set_title(
        f'SVD Spectrum of W ∈ ℝ^{{19×4096}}\n'
        f'Entropy rank: {svd_metrics["entropy_rank"]:.1f}/19  '
        f'  Cond κ: {svd_metrics["condition_number"]:.1f}',
        fontsize=10, fontweight='bold')

    # ── Panel 3: Principal angles ──────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.bar(range(1, len(angles)+1), angles, color='darkorange', alpha=0.8)
    ax3.axhline(45, color='gray',  linestyle='--', alpha=0.6, label='45° (random subspaces)')
    ax3.axhline(90, color='red',   linestyle='--', alpha=0.4, label='90° (orthogonal)')
    ax3.set_xlabel('Principal angle index', fontsize=9)
    ax3.set_ylabel('Angle (degrees)', fontsize=9)
    ax3.set_ylim(0, 95)
    ax3.legend(fontsize=8)
    ax3.set_title('Principal Angles: W Row Space vs PCA Subspace of H\n'
                  'θ = arccos(σᵢ of Q_Aᵀ Q_B),   0° = aligned,  90° = orthogonal',
                  fontsize=10, fontweight='bold')

    # ── Panel 4: Entanglement decomposition scatter ────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.scatter(cos_tri, sp_tri, alpha=0.45, s=28, color='purple', zorder=3)
    x_line = np.linspace(cos_tri.min() - 0.02, cos_tri.max() + 0.02, 200)
    ax4.plot(x_line, slope * x_line + intercept, 'r-', linewidth=2,
             label=f'y = {slope:.2f}x + {intercept:.2f}\nR² = {r2:.3f}')
    ax4.axhline(0, color='black', linewidth=0.5)
    ax4.axvline(0, color='black', linewidth=0.5)
    ax4.set_xlabel('Weight-space cosine similarity  cos(wᵢ, wⱼ)', fontsize=9)
    ax4.set_ylabel('Score-space Spearman correlation  ρ(sᵢ, sⱼ)', fontsize=9)
    ax4.set_title('Entanglement Decomposition\n'
                  'Architecture-driven vs data-driven correlation',
                  fontsize=10, fontweight='bold')
    ax4.legend(fontsize=9)

    plt.suptitle('ArmoRM Reward Head Geometry Audit\nW ∈ ℝ^{19×4096}',
                 fontsize=13, fontweight='bold')
    out = 'outputs/geometry_analysis.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved figure → {out}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n── Loading data ────────────────────────────────────────────────────────────")

    W = load_regression_weights()                            # (19, 4096)

    head_scores  = np.load('outputs/head_scores.npy')       # (800, 2, 19)
    hidden_states = np.load('outputs/hidden_states.npy')    # (800, 2, 4096)

    # Pool chosen and rejected into flat arrays
    scores_flat = head_scores.reshape(-1, N)                # (1600, 19)
    H_flat      = hidden_states.reshape(-1, 4096)           # (1600, 4096)

    # ── Analysis 1: Gram matrix / cosine similarity ──────────────────────────
    print("\n── Analysis 1: Gram matrix / cosine similarity (C = ŴŴᵀ) ─────────────────")
    C, G, orth_gap, norms = gram_analysis(W)

    print(f"  Weight vector norms  min={norms.min():.3f}  max={norms.max():.3f}  "
          f"mean={norms.mean():.3f}")
    print(f"  Orthogonality gap    ‖C − I‖_F / √(N(N−1)) = {orth_gap:.4f}")
    print(f"  (0 = perfectly orthogonal, 1 = maximally aligned)")

    idx = np.triu_indices(N, k=1)
    large_cos_pairs = [(HEAD_NAMES[i], HEAD_NAMES[j], C[i, j])
                       for i, j in zip(*idx) if abs(C[i, j]) > 0.25]
    large_cos_pairs.sort(key=lambda x: -abs(x[2]))
    print(f"\n  Weight-space pairs with |cos| > 0.25 ({len(large_cos_pairs)} pairs):")
    for a, b, v in large_cos_pairs[:10]:
        print(f"    {a:<26} ↔  {b:<26}  cos = {v:+.4f}")

    # ── Analysis 2: SVD / effective rank ─────────────────────────────────────
    print("\n── Analysis 2: SVD and effective rank ──────────────────────────────────────")
    U, S, Vt, svd_metrics = svd_analysis(W)

    print(f"  Singular values:\n  {np.array2string(S, precision=3, separator=', ')}")
    print(f"  Participation ratio   : {svd_metrics['participation_ratio']:.2f} / 19")
    print(f"  Entropy-based rank    : {svd_metrics['entropy_rank']:.2f} / 19")
    print(f"  Condition number κ    : {svd_metrics['condition_number']:.2f}")
    print(f"  k for 90% of W variance: {svd_metrics['k90']} singular vectors")

    # ── Analysis 3: Principal angles ─────────────────────────────────────────
    print("\n── Analysis 3: Principal angles (W row space vs PCA subspace of H) ────────")
    pca = PCA(n_components=N)
    pca.fit(H_flat)
    PCA_components = pca.components_    # (19, 4096) — top-19 PCA directions of H

    angles = principal_angles(W, PCA_components)
    print(f"  Principal angles (degrees):")
    print(f"  {np.array2string(angles, precision=1, separator=', ')}")
    print(f"  Mean: {angles.mean():.1f}°   Min: {angles.min():.1f}°   Max: {angles.max():.1f}°")
    print(f"  (0° = perfectly aligned with PCA subspace of H)")
    print(f"  (90° = orthogonal to PCA subspace of H)")

    # ── Analysis 4: Entanglement decomposition ────────────────────────────────
    print("\n── Analysis 4: Entanglement decomposition ──────────────────────────────────")
    C_weight, C_score, slope, intercept, r2, cos_tri, sp_tri = \
        entanglement_decomposition(W, scores_flat)

    print(f"  Linear regression across {len(cos_tri)} head pairs (upper triangle):")
    print(f"    Spearman(sᵢ, sⱼ) = {slope:.3f} × cos(wᵢ, wⱼ) + {intercept:.3f}")
    print(f"    R² = {r2:.4f}")
    print()
    if r2 < 0.25:
        print("  → R² is LOW: entanglement is predominantly DATA-DRIVEN.")
        print("    Weight vectors are near-orthogonal. The architecture supports independent heads.")
        print("    Heads are correlated because the training distribution conflates these concepts.")
        print("    Implication: targeted contrastive data could disentangle the heads.")
    elif r2 > 0.60:
        print("  → R² is HIGH: entanglement is predominantly ARCHITECTURE-DRIVEN.")
        print("    Weight vectors share directions. The model collapsed these concepts.")
        print("    Implication: orthogonality regularization on W is needed.")
    else:
        print("  → R² is MODERATE: entanglement has both data-driven and architecture-driven sources.")

    # ── Analysis 5: Projection variance ──────────────────────────────────────
    print("\n── Analysis 5: Projection variance ─────────────────────────────────────────")
    var_scoring, var_pca_r, rank_W = projection_variance(W, H_flat)

    print(f"  Rank of W: {rank_W}")
    print(f"  Variance in H captured by W's row space (S_W)  : {var_scoring*100:.2f}%")
    print(f"  Variance captured by top-{rank_W} PCA components of H : {var_pca_r*100:.2f}%")
    print(f"  Alignment gap: {(var_pca_r - var_scoring)*100:.2f} pp")
    print(f"  (Gap > 0 means scoring subspace misses some of the most variable directions in H)")

    # ── Save results ──────────────────────────────────────────────────────────
    results = {
        'orthogonality_gap'         : float(orth_gap),
        'svd_metrics'               : svd_metrics,
        'principal_angles_deg'      : angles.tolist(),
        'entanglement_r2'           : r2,
        'entanglement_slope'        : slope,
        'entanglement_intercept'    : intercept,
        'projection_var_scoring'    : float(var_scoring),
        'projection_var_pca'        : float(var_pca_r),
        'rank_W'                    : rank_W,
    }
    with open('outputs/geometry_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nSaved → outputs/geometry_results.json")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_all(C_weight, S, svd_metrics, angles, cos_tri, sp_tri, slope, intercept, r2)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n── Geometry Summary ─────────────────────────────────────────────────────────")
    print(f"  Orthogonality gap           : {orth_gap:.4f}   (0=orthogonal)")
    print(f"  Effective rank (entropy)    : {svd_metrics['entropy_rank']:.2f} / 19")
    print(f"  Condition number κ          : {svd_metrics['condition_number']:.2f}")
    print(f"  Mean principal angle        : {angles.mean():.1f}°")
    print(f"  Entanglement R²             : {r2:.4f}")
    print(f"  Projection variance (W)     : {var_scoring*100:.2f}%")
    print(f"  Projection variance (PCA)   : {var_pca_r*100:.2f}%")


if __name__ == '__main__':
    main()
