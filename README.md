# reward-basis-audit

ArmoRM has 19 labeled reward heads — helpfulness, safety, honesty, and 16 others. Do those labels correspond to what the model actually learned? I ran four interpretability analyses and a validity audit across three datasets to find out.

**Model:** RLHFlow/ArmoRM-Llama3-8B-v0.1  
**Datasets:** Anthropic/hh-rlhf, allenai/reward-bench, nvidia/HelpSteer2

**Full report available in REPORT.pdf**
---

## What I found

On hh-rlhf (naturalistic preference data), 1/19 heads predicts human judgment above chance. The valid one is readability (0.711). The helpfulness head scores 0.429 — inverted. Safety scores 0.520 — coin flip.

On RewardBench (capability-structured pairs), 17/19 heads are valid. Safety hits 0.851 overall, 0.95 on refusals-offensive specifically.

On HelpSteer2, I constructed pairs where exactly one dimension varies and the rest are held constant. Complexity goes from 0.380 on hh-rlhf to 0.857 on targeted pairs. Verbosity goes from 0.390 to 0.950. The model knew what these concepts meant — it just couldn't show it when everything co-varies at once.

The hh-rlhf failure is a measurement failure, not a model failure. But that distinction matters: a safety team auditing their reward model on naturalistic data would see the safety head at chance and draw the wrong conclusion.

The geometry analysis adds a secondary finding: the 19 weight vectors are nearly orthogonal (cosine gap 0.038), but behavioral correlation between heads is high (mean off-diagonal Spearman 0.454). Only 28% of the co-firing is explained by the weight geometry (R²=0.28). The entanglement lives in the backbone representations, not the architecture — safety and helpfulness co-vary in the hidden states because they co-vary in pretraining data.

---

## Files

```
extract_scores_and_embeddings.py   # run ArmoRM on hh-rlhf, save scores + hidden states
extract_rewardbench.py             # same on RewardBench (2985 pairs, 23 subsets)
extract_helpsteer2.py              # build TYPE A/B pairs from HelpSteer2
score_helpsteer2.py                # run ArmoRM on HelpSteer2 pairs
disentanglement_analysis.py        # pairwise Spearman between all 19 heads
linear_separability.py             # logistic regression probes from hidden states
geometry_analysis.py               # weight vector geometry, SVD, entanglement decomposition
activation_analysis.py             # token-level gradient attribution (written, deprioritized)
validity_check.py                  # preference accuracy + Wilcoxon on hh-rlhf
validity_check_rewardbench.py      # same, stratified by RewardBench subset
validity_check_helpsteer2.py       # TYPE A and TYPE B validity on HelpSteer2
utils.py                           # shared loading, scoring, saving
```

---

## Reproducing

```bash
pip install -r requirements.txt

# Step 1 
python extract_scores_and_embeddings.py

# Step 2 
python disentanglement_analysis.py
python linear_separability.py
python geometry_analysis.py
python validity_check.py

# Step 3 — RewardBench 
python extract_rewardbench.py
python validity_check_rewardbench.py

# Step 4 — HelpSteer2
python extract_helpsteer2.py
python score_helpsteer2.py
python validity_check_helpsteer2.py
```

