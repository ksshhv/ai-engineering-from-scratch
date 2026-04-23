# Differential Attention

> As of 2026-04-22. Softmax attention is a probability simplex. It has to put weight somewhere, so it puts a little weight everywhere -- and the "everywhere" is attention noise that leaks into the residual stream and blurs long-context retrieval. Differential attention builds two softmax maps and subtracts the second from the first, cancelling the common-mode noise the way a differential amplifier does. The needle survives. The distractors collapse toward zero. One published architecture, two papers, one subtraction.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 07 (attention from scratch), Phase 10 Lesson 04 (pre-training a mini GPT), Lesson 14 (open-model architecture knobs)
**Time:** ~45 minutes

## The Problem

Pick a long document, ask the model a question whose answer is one sentence buried in the middle, and plot the attention weights the model puts on each token. You will see a tall spike on the answer and a low, wide plateau on every other token. That plateau is the problem. Each position gets a few percent of the total mass. Multiplied by thousands of distractor tokens, it adds up to more attention mass than the answer token itself. The model is technically "attending" to the right place, but the noise floor is loud enough to corrupt the output.

This is not a tuning issue. It is baked into softmax. The function maps R^n onto the probability simplex, which means every output is strictly positive and the sum is exactly one. There is no way to say "zero attention" to a token. The best you can do is `1/n`, which for n=128000 is still a fat tail of common-mode noise. As context length grows, the tail grows. Long-context retrieval, long-document QA, and in-context learning all hit this wall.

The standard answer is "train bigger, use better data, sharpen the logits with temperature tricks." The Differential Transformer (Ye et al., arXiv:2410.05258) argues for a structural fix instead: if the problem is common-mode noise, use the same trick analog electronics has used for a century. Build two signal paths. Take the difference. The shared noise cancels. The unique signal doubles.

Concrete stakes. A 128k-context model with standard softmax attention spends a non-trivial fraction of its capacity compensating for its own attention noise. Every induction head has to fight the background. Every retrieval head has to out-shout the tail. The residual stream carries a blurred version of the target token mixed with weighted averages of distractors. Downstream layers then have to disambiguate the blur. Differential attention deletes that work by construction.

## The Concept

### Two softmax maps, one subtraction

Standard attention for one head:

```
A(Q, K) = softmax(Q KT / sqrt(d))
out = A V
```

Differential attention splits the Q and K projections into two halves:

```
A1 = softmax(Q1 K1T / sqrt(d))
A2 = softmax(Q2 K2T / sqrt(d))
A_diff = A1 - lambda * A2
out = A_diff V
```

`lambda` is a learned scalar, typically initialized around `0.8 - 0.6 * exp(-0.3 * (layer - 1))` so the subtraction is strongest near the top of the stack. The two projections Q1, K1 and Q2, K2 start at different random inits and diverge during training. The critical property is that both softmaxes place roughly the same low-mass background on distractor tokens -- that is the common-mode term. The signal token is the place where A1 puts lots of mass and A2 puts less. After subtraction, the signal survives and the background is pushed toward zero or slightly negative.

### Why subtraction expands the representation

A single softmax row is bounded in `(0, 1)` and has mass exactly one. Two subtracted softmaxes can produce any value in `(-lambda, 1)` and need not sum to any fixed number. That strictly expands the function class. The attention weights are no longer probabilities. They are scores. A negative score means "push this token away from the output" -- something a pure softmax head cannot express at all.

This is the same move that turned sigmoid outputs into GLU-style gates: trade "probability" for "signed score," gain representational room. The price is that the analytical interpretation of attention as a distribution is gone.

### The geometry in one picture

Think of each token's softmax row as a point on the simplex -- the triangle of all non-negative vectors that sum to one. Standard attention forces every row to live on this surface. Two different softmax heads live on two different points of the simplex, correlated because the underlying token geometry is the same. Their subtraction is a vector in the tangent space of the simplex -- not a probability, but a direction. Directions can point toward a token (positive weight), point away from a token (negative weight), or vanish (the two heads agreed on noise). That last case is the common-mode cancellation.

### Noise-canceling headphones, explicitly

The authors of the original DIFF Transformer paper draw the analogy to differential amplifiers and noise-canceling microphones. Two sensors pick up the same ambient noise plus one picks up the target signal. Subtracting the two cancels the noise and leaves the signal. The math is the same. The attention heads are the two sensors. The target signal is the needle token. The ambient noise is every other token fighting for mass in the softmax.

The analogy is tight enough to predict failure modes. A noise-canceling mic fails when the two microphones disagree on the noise (wind, sudden bursts, uncorrelated interference). Differential attention fails when the two softmax paths' distractor distributions diverge -- which is what tends to happen early in training before both projections have converged on a shared noise model. That is why DIFF V1's lambda schedule starts small at shallow layers and grows with depth: deeper layers have cleaner, more aligned noise to cancel.

### What it buys at scale

Reported gains from the paper and follow-ups (see Further Reading):

- Lower language modeling loss at matched parameter budgets.
- Better needle-in-haystack retrieval at long contexts, where standard attention's noise floor is worst.
- Reduced hallucination in long-context QA and summarization.
- Stronger in-context learning: cleaner induction heads.
- Smaller activation outliers, which makes downstream INT8 / FP8 quantization (Phase 10 Lesson 11) cleaner.
- Sparser attention maps without an explicit sparsity loss. Lower attention entropy falls out of the subtraction itself.

The gains on short-context chat are modest. The payoff is concentrated exactly where softmax's noise floor hurts the most. Reported scaling is roughly "a differential transformer with N parameters matches a standard transformer with 1.5-2x N parameters on long-context retrieval evals," though the exact multiplier depends on the eval and model size.

### The V2 rework (Microsoft, Jan 2026)

Ye, Dong, Sun, and Wei shipped Differential Transformer V2 in January 2026. Three changes matter for implementers:

1. Q2 gets its own parameters (borrowed from elsewhere in the block) instead of splitting the baseline head in half. The head-size matches the baseline, so FlashAttention works unmodified -- no custom kernel.
2. The per-head RMSNorm after the subtraction is removed. Microsoft observed it caused loss and gradient spikes at 7B+ pretraining scale.
3. The lambda parameterization is simplified and the initialization is changed. DIFF V1's globally shared scheduled lambda is gone.

The functional form -- two softmax maps, subtract with lambda -- is unchanged. Everything else is training-stability plumbing. If you are writing the architecture today, you almost always want V2.

### The lineage

Differential attention is one node in a small family. Grouped Differential Attention (GDA) trades balanced head allocation for asymmetric groups, assigning more heads to signal extraction and fewer to noise control, echoing the GQA trick from Lesson 14. Differential Gated Self-Attention replaces the single scalar lambda with a per-token gate. None of these follow-ups has displaced the original at scale yet; DIFF V2 is the current reference implementation. The pattern is worth watching: whenever a single scalar mediates a subtraction, someone will replace it with a learned gate. The question is whether the extra parameters pay for themselves at pretraining scale.

## Build It

The minimal implementation is fifty lines. Two Q projections, two K projections, two softmaxes, subtract. See `code/main.py` for the runnable version.

### Step 1: one head, standard attention

Run a single head on an 8-token sequence. Build one signal direction and one noise direction, then make six of the eight tokens point along the noise direction (distractors), token 5 along the signal (the needle), and token 7 along the signal (the query). Standard softmax puts a tall spike on the needle but also gives each distractor a few percent.

### Step 2: split Q and K, build two maps

Project the query token twice, the key tokens twice, independently. Compute two softmax attention vectors. Both vectors put most mass on the needle, but they disagree on the exact per-distractor distribution because their projection weights are independent.

### Step 3: subtract with lambda

Compute `A_diff = A1 - lambda * A2`. Lambda = 0.7 in the demo. The needle entry stays large and positive. The distractor entries, which are small and roughly equal in both maps, collapse toward zero or go slightly negative. That is the noise cancellation.

### Step 4: measure

Report needle mass, distractor mass, entropy, and the needle-to-noise ratio. The runnable script on this lesson's seed (20260422) produces:

```
standard     needle=0.288  distractors=+0.476  SNR=0.60
differential needle=0.118  distractors=+0.091  SNR=1.29
```

Differential attention shrank the needle slightly but shrank the distractor mass by more than 5x, lifting the signal-to-noise ratio above 1 for the first time. Exactly the shape the paper promises.

Two things to notice. First, the entropy drops: 1.90 nats to 0.83 nats. A lower-entropy attention map is a sharper attention map, which is the paper's other headline claim -- differential attention produces sparse attention patterns without any sparsity regularizer. Second, the absolute needle mass is lower with differential attention than with standard attention, but the *relative* mass is higher. What matters for the residual stream is the signed weighted sum of value vectors. If distractors are near zero or negative, they stop leaking into the output no matter how many of them there are.

### Step 5: why this is not the same as temperature

An obvious alternative is to sharpen the standard softmax with a low temperature. That kills distractor mass too. The difference: temperature sharpening is a fixed, parameter-free transform. It cannot learn which tokens are distractors. Differential attention learns both softmax maps jointly during pretraining; the second map is trained to be the noise estimate. The model discovers what its own attention noise looks like and subtracts it. Temperature cannot.

## Use It

A production-grade implementation adds three things to this toy.

First, multi-head plumbing. Each head has its own `(Q1, K1, Q2, K2, V)` projections. Heads are averaged after per-head RMSNorm (DIFF V1) or concatenated directly (DIFF V2). The overall block has the same FLOPs as a standard attention block with the same head count.

Second, the lambda schedule. DIFF V1 initializes lambda per-layer as a function of depth so early layers subtract less. DIFF V2 picks a simpler scheme. In both cases, lambda is a learned scalar that the optimizer shifts during training.

Third, kernel fusion. The naive implementation stores two full `N x N` attention maps, which doubles activation memory for long contexts. DIFF V1 ships a custom kernel. DIFF V2 uses standard FlashAttention twice and subtracts the outputs, which is the unlock that makes the approach practical for 7B+ pretraining.

When you integrate this on top of the Lesson 14 architecture knobs, differential attention replaces the attention-map step regardless of whether the underlying attention is MHA, GQA, or MLA. The six knobs from Lesson 14 still apply. Differential attention is a seventh knob -- smaller gain than RoPE or MoE but a genuine improvement on long-context and retrieval tasks.

### When it helps and when it does not

Pick differential attention when the target workload is *search-shaped*: long documents, retrieval, in-context learning with many exemplars, multi-hop QA, or any case where the model's job is to find a specific token in a sea of distractors. Those are exactly the workloads where softmax's noise floor dominates the error budget.

Skip it when the workload is *generation-shaped* on short contexts: conversational chat, short-form creative writing, instruction following on single-turn prompts. Standard softmax attention already concentrates mass where it needs to, the noise floor is small in absolute terms, and the extra projections are parameter budget you could spend elsewhere (wider hidden size, more layers, more training tokens).

Skip it also when the model is already trained. Differential attention is not drop-in. The second softmax path needs to be trained from initialization to become a noise estimator; bolting it onto a pretrained checkpoint and running a few thousand steps of finetuning does not reproduce the published gains.

### Interaction with quantization

One underrated result in the original paper is that differential attention reduces the magnitude of activation outliers. Activation outliers are the tokens that produce hundred-sigma values in intermediate tensors, and they are the single biggest obstacle to aggressive INT8 or FP8 quantization (see Phase 10 Lesson 11). Sparser, signed attention weights mean the output of the attention block has a tighter distribution. The knock-on effect is smoother post-training quantization, fewer calibration outliers, and tighter FP8 scale factors. If your serving stack is FP8 or INT8, differential attention is worth the experiment cost even if the language modeling loss looks flat.

## Ship It

This lesson's artifact is `outputs/skill-diff-attn.md` -- a decision skill for when to adopt differential attention. Given a model spec, a task profile, and a baseline attention choice, it returns a recommendation with explicit ablation requirements, lambda init, norm placement, and kernel compatibility. The skill refuses to drop differential attention into an existing pretrained checkpoint without finetuning, and refuses to recommend DIFF V1 at 7B+ scale without warning about the reported instability.

The decision tree inside the skill is short: task is retrieval / long-context / ICL-heavy, budget includes a full pretrain or extended continued pretrain, target stack supports FlashAttention. Three yeses and differential attention is on the shortlist. One no and you stick with the Lesson 14 baseline. The skill also forces an ablation budget: a 350M-1.3B matched-parameter run against a standard-attention baseline on the target data mix plus a needle-in-haystack eval. Without those numbers the recommendation is not actionable.

## Exercises

1. Swap the seed in `code/main.py` to 42 and re-run. Does differential attention still beat standard attention on signal-to-noise? If not, what does that say about the sensitivity of the mechanism to projection initialization?

2. Sweep lambda from 0.1 to 1.2 in steps of 0.1. Plot needle mass, distractor mass, and signal-to-noise. Identify the lambda value where distractor mass first crosses zero. Explain why the optimum is not at lambda = 1.

3. Increase `SEQ_LEN` from 8 to 64 and reshape the signal so only one token is the needle. Standard softmax's noise floor should get worse (more distractors, same total mass). Differential attention should hold up. Measure the SNR gap as a function of sequence length.

4. Modify the build to use two heads and concatenate their differential outputs. Does the needle mass add up linearly, or do the heads specialize? This is a miniature version of the multi-head differential attention experiment from Ye et al.

5. Re-read the DIFF V2 blog post. Implement the V2 simplification to the lambda parameterization on top of your multi-head version. Train a 10k-step character-level LM on a small corpus (reuse the Lesson 04 data pipeline) and report whether DIFF V2 matches or beats a standard attention baseline at the same parameter count.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Differential attention | "Two heads minus each other" | Split Q and K in half, compute two softmax maps, subtract the second from the first with a learned scalar lambda — common-mode noise cancels |
| Attention noise | "The model attends to junk" | The small but nonzero probability mass softmax puts on every non-relevant token — scales with sequence length, corrupts long-context retrieval |
| Common-mode noise | "Noise shared across paths" | The portion of the attention distribution that is roughly identical in two independently projected softmax maps — what the subtraction cancels |
| Lambda (differential) | "The mixing weight" | A learned scalar that weights the second softmax before subtraction — around 0.6-0.8 in practice, initialized with a depth schedule in V1 |
| DIFF Transformer | "Microsoft's noise-canceling LLM" | The architecture built on differential attention (Ye et al., 2024) — matched baseline params by splitting heads in half |
| DIFF V2 | "The stable one" | 2026 rework that adds Q2 params instead of splitting, drops the per-head RMSNorm, simplifies lambda — FlashAttention-compatible, stable at 7B+ |
| Needle-in-haystack | "Find this fact in a long doc" | A benchmark category where softmax's noise floor hurts most — differential attention's reported gains are largest here |
| Negative attention weight | "Push this token away" | A capability standard softmax does not have — differential attention's subtraction allows weights below zero for true suppression |

## Further Reading

- [Ye, Dong, Xia, Sun, Zhu, Huang, Wei — "Differential Transformer" (arXiv:2410.05258, Oct 2024; ICLR 2025)](https://arxiv.org/abs/2410.05258) — the original paper. Read sections 2 (mechanism) and 4 (experiments); the needle-in-haystack plot is the canonical motivation.
- [Ye, Dong, Sun, Wei — "Differential Transformer V2" (Microsoft Research blog, Jan 2026)](https://huggingface.co/blog/microsoft/diff-attn-v2) — the training-stability rework. Read before picking V1 vs V2 for any real training run.
- [Shazeer — "GLU Variants Improve Transformer" (arXiv:2002.05202)](https://arxiv.org/abs/2002.05202) — background on why gated / signed activations beat unsigned ones, which is the same bet differential attention makes on the attention weights themselves.
- [Su et al. — "RoFormer: Enhanced Transformer with Rotary Position Embedding" (arXiv:2104.09864)](https://arxiv.org/abs/2104.09864) — long-context attention prerequisite from Lesson 14; differential attention's gains compound with long-context position schemes.
- [Ainslie et al. — "GQA: Training Generalized Multi-Query Transformer Models" (arXiv:2305.13245)](https://arxiv.org/abs/2305.13245) — context for how attention heads already share structure in modern transformers; differential attention is orthogonal to GQA and stacks on top of it.
- [Vaswani et al. — "Attention Is All You Need" (arXiv:1706.03762)](https://arxiv.org/abs/1706.03762) — the original softmax attention this lesson argues against. Worth a re-read as the single-path baseline every differential-attention experiment is measured against.
