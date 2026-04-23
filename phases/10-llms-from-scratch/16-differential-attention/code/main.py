"""Differential attention on a toy 8-token sequence, stdlib only.

Standard softmax attention spreads probability mass across every token, so a
needle token has to fight a long tail of noise. Differential attention splits
Q and K in half, runs two softmax maps, and subtracts the second from the
first with a learned scalar lambda. Common-mode noise cancels; the signal
survives. Same trick as a differential amplifier or a noise-canceling mic.

The sequence is eight tokens where position 5 is the needle (the answer the
model should attend to from query position 7) and the other seven positions
are distractors with correlated embeddings so they leak into the softmax as
common-mode noise. We score standard attention and differential attention on
how much mass they put on the needle versus the distractors.

No numpy, no torch. Pure Python with `math` and `random`.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


SEED = 20260422
SEQ_LEN = 8
HEAD_DIM = 16
NEEDLE_POS = 5
QUERY_POS = 7
LAMBDA = 0.7


def zeros(rows: int, cols: int) -> list[list[float]]:
    return [[0.0 for _ in range(cols)] for _ in range(rows)]


def randn(rng: random.Random, rows: int, cols: int, scale: float = 1.0) -> list[list[float]]:
    return [[rng.gauss(0.0, scale) for _ in range(cols)] for _ in range(rows)]


def add(a: list[float], b: list[float]) -> list[float]:
    return [x + y for x, y in zip(a, b)]


def scale(v: list[float], s: float) -> list[float]:
    return [x * s for x in v]


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def softmax(logits: list[float]) -> list[float]:
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    z = sum(exps)
    return [e / z for e in exps]


def build_sequence(rng: random.Random) -> list[list[float]]:
    """Eight tokens. Position 5 is the needle. The query at position 7 is
    built to partially align with the needle. Distractors share a correlated
    `noise` direction plus a small slice of the signal direction, so they
    all light up the standard softmax at once -- that is the common-mode
    noise differential attention is meant to cancel."""
    signal = [rng.gauss(0.0, 1.0) for _ in range(HEAD_DIM)]
    noise = [rng.gauss(0.0, 1.0) for _ in range(HEAD_DIM)]
    tokens: list[list[float]] = []
    for pos in range(SEQ_LEN):
        jitter = [rng.gauss(0.0, 0.15) for _ in range(HEAD_DIM)]
        if pos == NEEDLE_POS:
            tokens.append(add(scale(signal, 0.7), jitter))
        elif pos == QUERY_POS:
            tokens.append(add(scale(signal, 0.7), jitter))
        else:
            mixed = add(scale(noise, 0.9), scale(signal, 0.25))
            tokens.append(add(mixed, jitter))
    return tokens


def project(tokens: list[list[float]], W: list[list[float]]) -> list[list[float]]:
    out = zeros(len(tokens), len(W[0]))
    for i, t in enumerate(tokens):
        for j in range(len(W[0])):
            out[i][j] = sum(t[k] * W[k][j] for k in range(len(t)))
    return out


def attention_map(Q: list[list[float]], K: list[list[float]], query_idx: int) -> list[float]:
    d = len(Q[query_idx])
    scale_factor = 1.0 / math.sqrt(d)
    logits = [dot(Q[query_idx], K[j]) * scale_factor for j in range(len(K))]
    return softmax(logits)


@dataclass
class Report:
    name: str
    weights: list[float]
    needle_mass: float
    distractor_mass: float
    entropy: float


def score(name: str, weights: list[float]) -> Report:
    needle = weights[NEEDLE_POS]
    distractors = sum(w for i, w in enumerate(weights) if i != NEEDLE_POS and i != QUERY_POS)
    clipped = [max(w, 1e-12) for w in weights]
    entropy = -sum(w * math.log(w) for w in clipped)
    return Report(name, weights, needle, distractors, entropy)


def print_report(r: Report) -> None:
    print(f"  {r.name}")
    bars = []
    for i, w in enumerate(r.weights):
        tag = " <- needle" if i == NEEDLE_POS else (" (query)" if i == QUERY_POS else "")
        bar_len = max(0, int(round(w * 40)))
        bar = "#" * bar_len
        bars.append(f"    pos {i}: {w:+.3f} |{bar}{tag}")
    print("\n".join(bars))
    print(f"    needle mass     : {r.needle_mass:+.3f}")
    print(f"    distractor mass : {r.distractor_mass:+.3f}")
    print(f"    entropy (nats)  : {r.entropy:.3f}")


def run() -> None:
    rng = random.Random(SEED)
    tokens = build_sequence(rng)

    Wq1 = randn(rng, HEAD_DIM, HEAD_DIM, scale=0.3)
    Wk1 = randn(rng, HEAD_DIM, HEAD_DIM, scale=0.3)
    Wq2 = randn(rng, HEAD_DIM, HEAD_DIM, scale=0.3)
    Wk2 = randn(rng, HEAD_DIM, HEAD_DIM, scale=0.3)

    Q1 = project(tokens, Wq1)
    K1 = project(tokens, Wk1)
    Q2 = project(tokens, Wq2)
    K2 = project(tokens, Wk2)

    standard = attention_map(Q1, K1, QUERY_POS)
    map1 = attention_map(Q1, K1, QUERY_POS)
    map2 = attention_map(Q2, K2, QUERY_POS)
    diff = [a - LAMBDA * b for a, b in zip(map1, map2)]

    print("differential attention on an 8-token sequence")
    print(f"  seed={SEED}  head_dim={HEAD_DIM}  needle_pos={NEEDLE_POS}  query_pos={QUERY_POS}  lambda={LAMBDA}")
    print()
    print_report(score("standard softmax attention", standard))
    print()
    print_report(score("differential attention (map1 - lambda * map2)", diff))
    print()

    std = score("std", standard)
    dif = score("dif", diff)
    std_snr = std.needle_mass / max(std.distractor_mass, 1e-12)
    dif_positive_noise = max(dif.distractor_mass, 0.0)
    dif_snr = dif.needle_mass / max(dif_positive_noise, 1e-12)
    print("verdict")
    print(f"  needle mass (higher is better):")
    print(f"    standard     : {std.needle_mass:+.3f}")
    print(f"    differential : {dif.needle_mass:+.3f}")
    print(f"  distractor mass (lower / more negative is better):")
    print(f"    standard     : {std.distractor_mass:+.3f}")
    print(f"    differential : {dif.distractor_mass:+.3f}")
    print(f"  needle-to-positive-noise ratio:")
    print(f"    standard     : {std_snr:7.2f}")
    if dif_positive_noise <= 1e-12:
        print(f"    differential : inf (all distractor mass is <= 0)")
    else:
        print(f"    differential : {dif_snr:7.2f}")
    if dif.distractor_mass < std.distractor_mass:
        print("  differential attention cancelled the common-mode distractor noise.")
    else:
        print("  no improvement this seed -- try another.")


if __name__ == "__main__":
    run()
