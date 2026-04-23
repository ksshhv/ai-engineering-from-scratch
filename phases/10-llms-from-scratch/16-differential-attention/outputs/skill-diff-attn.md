---
name: diff-attn
description: Decide whether to replace standard softmax attention with differential attention for a given model and task, and configure it correctly.
version: 1.0.0
phase: 10
lesson: 16
tags: [attention, differential-attention, diff-transformer, long-context, hallucination, retrieval]
---

Given a model spec (parameter count, training budget, target context length, target tasks) and an attention baseline (MHA / GQA / MLA), decide whether to adopt differential attention (Ye et al., arXiv:2410.05258; DIFF V2, Microsoft, Jan 2026) and how to configure it.

Produce:

1. Fit check. Name the task profile (key-information retrieval, long-document QA, in-context learning, code reasoning, hallucination-prone generation, chat). Differential attention has the strongest published gains on the first three; flag gracefully if the target task is generic chat where the lift is smaller.

2. Head budget. Original DIFF splits each head in half so parameter count matches the baseline. DIFF V2 adds a separate Q2 projection by borrowing parameters from elsewhere, trading a small parameter bump for FlashAttention compatibility. Pick one and justify.

3. Lambda schedule. DIFF V1 uses a globally shared learned lambda with a depth-dependent initialization. DIFF V2 simplifies parameterization. Specify the initial value, whether it is shared across heads, and whether it is per-layer or global.

4. Norm placement. DIFF V1 applies per-head RMSNorm after the subtraction. DIFF V2 removes it after finding it unstable at LLM scale. Pick based on target parameter count: keep the per-head norm for small-scale experiments (< 3B), drop it for 7B+ pretraining.

5. Kernel and memory. State whether the chosen variant is FlashAttention-compatible. Standard DIFF V1 is not (it needs a custom kernel for the two-softmax-then-subtract pattern). DIFF V2 is. Reject the recommendation if target serving stack is locked to FlashAttention and you picked V1.

6. Ablation gate. Propose the minimum ablation before full pretraining: small-scale (350M - 1.3B) LM loss against a matched-parameter baseline on the target data mix, plus a needle-in-haystack eval at the target context length. Only scale up after DIFF wins both.

Hard rejects:
- Recommending DIFF V1 on a 7B+ pretraining run without warning about reported loss spikes and gradient instability.
- Recommending differential attention on top of a fixed off-the-shelf checkpoint without finetuning. Differential attention must be trained from initialization or via extended finetuning; it is not drop-in on pretrained weights.
- Quoting published gains without specifying which task category they came from (the lift on long-context retrieval is larger than the lift on generic LM perplexity).
- Conflating differential attention (two softmax maps, subtract) with differential privacy (noise addition for privacy). They are unrelated.

Output: a one-page recommendation with model config diff (new fields, removed fields), lambda init value, norm choice, kernel choice, ablation plan with specific eval names (MMLU, Needle, LongBench, TriviaQA with long context), and a "worth reconsidering if..." paragraph naming the specific training signal (loss spike, needle-eval regression, throughput cliff) that would flip the choice back to a standard attention baseline.
