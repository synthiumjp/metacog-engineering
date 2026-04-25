"""
Shared utilities for Phase 0 Steps 1, 1b, 2, 3, 4.

Provides:
    - parse_response: extract (answer, confidence) from model output
    - extract_hidden_state: pull last-layer hidden state at answer-token position
    - paired_bootstrap_auroc2_delta: paired bootstrap CI on AUROC₂ delta
    - vrs_screen: compute L, Fp, RBS, TRIN, r(confidence, correct), tier label
    - is_correct: TriviaQA-aliases correctness check

All functions are pure where possible. Hidden-state extraction requires a live
model and is generator-style to keep VRAM under control.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

_CONFIDENCE_PATTERNS = [
    # "Confidence: 75%", "Confidence: 75", "confidence 75%"
    re.compile(r"confidence\s*:?\s*(\d{1,3})\s*%?", re.IGNORECASE),
    # Bare percentage anywhere in the reply after the answer
    re.compile(r"(\d{1,3})\s*%"),
    # Trailing bare integer in [0, 100]
    re.compile(r"\b(\d{1,3})\b\s*$"),
]


@dataclass
class ParsedResponse:
    raw: str
    answer: str           # the answer text; may be empty string if unparseable
    confidence: float     # in [0, 100]; np.nan if unparseable
    answer_token_offset: int  # index of the answer's first token relative to
                              # the start of generated tokens; -1 if unknown


def parse_response(
    raw: str,
    prompt_format: str = "triviaqa",
) -> ParsedResponse:
    """Parse a generated response into (answer, confidence).

    Heuristics:
        - TriviaQA: answer is first non-empty line, strip leading "Answer:"
        - MMLU: answer is first letter (A/B/C/D) appearing after "Answer:"
        - Confidence is the first integer 0-100 matched by any of the patterns
          above, searched in order of specificity.
    """
    text = raw.strip()
    answer = ""
    confidence = float("nan")

    if prompt_format == "triviaqa":
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        if lines:
            first = lines[0]
            m = re.match(r"^(?:answer\s*:?\s*)(.*)", first, flags=re.IGNORECASE)
            answer = m.group(1).strip() if m else first
            # Strip trailing "Confidence: X%" from the answer if it leaked in
            answer = re.sub(
                r"\s*[,;]?\s*confidence\s*:?.*$", "", answer, flags=re.IGNORECASE
            ).strip()
    elif prompt_format == "mmlu":
        m = re.search(r"answer\s*:?\s*([A-D])", text, flags=re.IGNORECASE)
        if m:
            answer = m.group(1).upper()
    else:
        raise ValueError(f"Unknown prompt_format: {prompt_format}")

    for pat in _CONFIDENCE_PATTERNS:
        m = pat.search(text)
        if m:
            try:
                v = int(m.group(1))
                if 0 <= v <= 100:
                    confidence = float(v)
                    break
            except ValueError:
                continue

    return ParsedResponse(
        raw=raw,
        answer=answer,
        confidence=confidence,
        answer_token_offset=-1,  # set by caller if computed from generation
    )


# ---------------------------------------------------------------------------
# Correctness (TriviaQA)
# ---------------------------------------------------------------------------

_ARTICLE_RE = re.compile(r"^(the|a|an)\s+")
_PUNCT_RE = re.compile(r"[^\w\s]")
_WS_RE = re.compile(r"\s+")


def _normalise(s: str) -> str:
    s = s.lower().strip()
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    s = _ARTICLE_RE.sub("", s)
    return s


def is_correct_triviaqa(pred: str, aliases: Sequence[str]) -> bool:
    """TriviaQA correctness: alias appears in pred (or vice versa) after norm."""
    p = _normalise(pred)
    if not p:
        return False
    for a in aliases:
        an = _normalise(a)
        if not an:
            continue
        if an in p or p in an:
            return True
    return False


def is_correct_mmlu(pred: str, gold: str) -> bool:
    """MMLU correctness: exact letter match after uppercasing."""
    return pred.strip().upper() == gold.strip().upper()


# ---------------------------------------------------------------------------
# Hidden-state extraction
# ---------------------------------------------------------------------------

def find_answer_token_positions(
    tokenizer,
    full_ids: "np.ndarray | list[int]",
    prompt_len: int,
    answer_text: str,
) -> tuple[int, int]:
    """Locate the answer-token span within a generated sequence.

    Returns (start_idx, end_idx) as offsets relative to the *start of generation*
    (i.e., relative to full_ids[prompt_len:]). Returns (-1, -1) on failure.

    Strategy: tokenise answer_text on its own and find its first subsequence
    match in the generated tokens. This is robust to BPE quirks because we
    compare token-id sequences, not strings.
    """
    gen_ids = list(full_ids[prompt_len:])
    if not answer_text:
        return (-1, -1)

    # Gemma tokenisers typically prepend a space boundary; try both variants.
    candidates = [answer_text, " " + answer_text]
    for cand in candidates:
        ans_ids = tokenizer(cand, add_special_tokens=False).input_ids
        if not ans_ids:
            continue
        n = len(ans_ids)
        for i in range(len(gen_ids) - n + 1):
            if gen_ids[i : i + n] == ans_ids:
                return (i, i + n)
    return (-1, -1)


def extract_answer_token_hidden_state(
    hidden_states_stack: "np.ndarray",
    answer_start: int,
    answer_end: int,
    position: str = "last_answer_token",
) -> np.ndarray:
    """Extract a single hidden state from a stack.

    Args:
        hidden_states_stack: shape (n_generated_tokens, hidden_dim), last layer only
        answer_start, answer_end: offsets into generated tokens
        position: one of:
            "first_answer_token"  -> hidden_states_stack[answer_start]
            "last_answer_token"   -> hidden_states_stack[answer_end - 1]
            "pre_answer_token"    -> hidden_states_stack[answer_start - 1]
    """
    if answer_start < 0 or answer_end <= answer_start:
        raise ValueError("Invalid answer span")
    if position == "first_answer_token":
        return hidden_states_stack[answer_start]
    if position == "last_answer_token":
        return hidden_states_stack[answer_end - 1]
    if position == "pre_answer_token":
        idx = max(0, answer_start - 1)
        return hidden_states_stack[idx]
    raise ValueError(f"Unknown position: {position}")


# ---------------------------------------------------------------------------
# AUROC₂ and paired bootstrap
# ---------------------------------------------------------------------------

def auroc2(confidence: np.ndarray, correct: np.ndarray) -> float:
    """Compute AUROC₂ (ROC AUC of confidence vs binary correctness).

    Ties in confidence count as incorrect per the CMM convention
    (the smaller-signed score when correctness is 1 is treated as loss);
    we use sklearn's default which handles ties via the probabilistic rule,
    close enough for Phase 0. If a stricter tie convention is needed,
    replace with a manual Mann-Whitney U implementation.
    """
    from sklearn.metrics import roc_auc_score

    c = np.asarray(confidence, dtype=float)
    y = np.asarray(correct, dtype=int)
    mask = ~np.isnan(c)
    if mask.sum() < 2:
        return float("nan")
    if y[mask].sum() == 0 or y[mask].sum() == mask.sum():
        # Degenerate: all correct or all incorrect — AUROC₂ undefined.
        return float("nan")
    return float(roc_auc_score(y[mask], c[mask]))


def paired_bootstrap_auroc2_delta(
    confidence_a: np.ndarray,
    confidence_b: np.ndarray,
    correct: np.ndarray,
    n_resamples: int = 10_000,
    seed: int = 42,
    ci: float = 0.95,
) -> dict:
    """Paired bootstrap CI on (AUROC₂_a - AUROC₂_b).

    Both confidence vectors are evaluated on the same items (correct[i] is the
    same label for both). At each resample, draw item indices with replacement,
    compute AUROC₂ for each condition on that index set, take the difference.

    Returns a dict with:
        point_a, point_b, point_delta: AUROC₂ values on full sample
        lo, hi: CI bounds on the delta
        n_resamples, seed, ci
    """
    rng = np.random.default_rng(seed)
    a = np.asarray(confidence_a, dtype=float)
    b = np.asarray(confidence_b, dtype=float)
    y = np.asarray(correct, dtype=int)
    n = len(y)
    assert len(a) == n and len(b) == n, "Mismatched lengths"

    point_a = auroc2(a, y)
    point_b = auroc2(b, y)
    point_delta = point_a - point_b

    deltas = np.empty(n_resamples, dtype=float)
    valid = 0
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        ya = y[idx]
        if ya.sum() == 0 or ya.sum() == n:
            deltas[i] = np.nan
            continue
        da = auroc2(a[idx], ya)
        db = auroc2(b[idx], ya)
        deltas[i] = da - db
        valid += 1

    deltas = deltas[~np.isnan(deltas)]
    alpha = (1 - ci) / 2
    lo = float(np.quantile(deltas, alpha))
    hi = float(np.quantile(deltas, 1 - alpha))

    return {
        "point_a": point_a,
        "point_b": point_b,
        "point_delta": point_delta,
        "lo": lo,
        "hi": hi,
        "ci": ci,
        "n_resamples": n_resamples,
        "n_valid": valid,
        "seed": seed,
    }


# ---------------------------------------------------------------------------
# VRS screen (Validity of Rating Scale — CMM programme convention)
# ---------------------------------------------------------------------------

def vrs_screen(
    confidence: np.ndarray,
    correct: np.ndarray,
    ceiling_threshold: float = 95.0,
    floor_threshold: float = 5.0,
) -> dict:
    """Compute the Validity of Rating Scale screen used in the CMM programme.

    Indices:
        L   — ceiling rate: P(confidence >= ceiling_threshold)
        Fp  — floor rate: P(confidence <= floor_threshold)
        RBS — rating-bin sparsity: 1 - (number of distinct bins used / total possible)
        TRIN — true response invariance: P(modal confidence class)
        r   — Pearson r(confidence, correct)

    Tier classification:
        Invalid:        L >= 0.70 OR TRIN >= 0.80 OR |r| < 0.05
        Indeterminate:  0.40 <= L < 0.70 OR 0.60 <= TRIN < 0.80
        Valid:          otherwise

    Thresholds are Phase 0 defaults consistent with the saturation paper;
    do not re-tune.
    """
    c = np.asarray(confidence, dtype=float)
    y = np.asarray(correct, dtype=int)
    mask = ~np.isnan(c)
    c, y = c[mask], y[mask]
    n = len(c)
    if n == 0:
        return {"tier": "undefined", "n": 0}

    L = float(np.mean(c >= ceiling_threshold))
    Fp = float(np.mean(c <= floor_threshold))

    # RBS: bin confidence into deciles; count distinct bins
    bins = np.clip((c // 10).astype(int), 0, 10)
    distinct = len(np.unique(bins))
    RBS = 1 - distinct / 11

    # TRIN: modal bin share
    unique, counts = np.unique(bins, return_counts=True)
    TRIN = float(counts.max() / counts.sum())

    if c.std() == 0 or y.std() == 0:
        r = 0.0
    else:
        r = float(np.corrcoef(c, y)[0, 1])

    if L >= 0.70 or TRIN >= 0.80 or abs(r) < 0.05:
        tier = "Invalid"
    elif L >= 0.40 or TRIN >= 0.60:
        tier = "Indeterminate"
    else:
        tier = "Valid"

    return {
        "L": L, "Fp": Fp, "RBS": RBS, "TRIN": TRIN, "r": r,
        "tier": tier, "n": n,
    }


# ---------------------------------------------------------------------------
# Logit entropy (for E5, computed at Step 1 time)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Disjointness: saturation paper exclusion + TriviaQA pool partitioning
# ---------------------------------------------------------------------------

# Total items in TriviaQA rc.nocontext validation split (HuggingFace).
TRIVIAQA_VAL_SIZE = 17_944

# Saturation paper draw parameters (deterministic reproduction).
SATURATION_SEED = 42
SATURATION_N = 524


def get_saturation_exclusion_indices() -> set[int]:
    """Reproduce the exact 524 dataset indices used in the saturation paper.

    The saturation paper's ``collect_saturation.py`` draws::

        rng = np.random.default_rng(seed=42)
        indices = rng.choice(17944, size=524, replace=False).tolist()

    This function replicates that draw so disjointness can be enforced
    programmatically without depending on an external manifest file.
    """
    rng = np.random.default_rng(SATURATION_SEED)
    return set(rng.choice(TRIVIAQA_VAL_SIZE, size=SATURATION_N, replace=False).tolist())


def partition_triviaqa_pool(
    seed: int = 42,
    n_teval: int = 1_000,
    n_tcal: int = 2_000,
    n_step0: int = 500,
) -> dict[str, list[int]]:
    """Partition the TriviaQA validation split into disjoint index sets.

    Steps:
        1. Exclude the saturation paper's 524 indices.
        2. Shuffle the remaining pool with ``random.Random(seed)``.
        3. Slice contiguously: T-eval, T-cal, Step 0.

    Returns a dict with keys ``teval``, ``tcal``, ``step0``, each mapping
    to a sorted list of *dataset* indices (positions in the HF dataset).
    Also includes ``saturation`` for cross-check logging.
    """
    import random

    sat_indices = get_saturation_exclusion_indices()
    pool = sorted(set(range(TRIVIAQA_VAL_SIZE)) - sat_indices)
    assert len(pool) == TRIVIAQA_VAL_SIZE - len(sat_indices)

    rng = random.Random(seed)
    rng.shuffle(pool)

    total_needed = n_teval + n_tcal + n_step0
    if len(pool) < total_needed:
        raise RuntimeError(
            f"After excluding {len(sat_indices)} saturation items, "
            f"pool has {len(pool)} items but {total_needed} are needed "
            f"({n_teval} + {n_tcal} + {n_step0})."
        )

    teval = pool[:n_teval]
    tcal = pool[n_teval : n_teval + n_tcal]
    step0 = pool[n_teval + n_tcal : n_teval + n_tcal + n_step0]

    # Sanity: pairwise disjointness
    sets = {"teval": set(teval), "tcal": set(tcal),
            "step0": set(step0), "saturation": sat_indices}
    for a_name, a_set in sets.items():
        for b_name, b_set in sets.items():
            if a_name >= b_name:
                continue
            overlap = a_set & b_set
            if overlap:
                raise RuntimeError(
                    f"Disjointness violation: {a_name} ∩ {b_name} = "
                    f"{len(overlap)} items (e.g. {sorted(overlap)[:5]})"
                )

    return {
        "teval": sorted(teval),
        "tcal": sorted(tcal),
        "step0": sorted(step0),
        "saturation": sorted(sat_indices),
    }


# ---------------------------------------------------------------------------
# Logit entropy (for E5, computed at Step 1 time)
# ---------------------------------------------------------------------------

def shannon_entropy_from_logits(logits: np.ndarray) -> float:
    """Compute Shannon entropy (natural log) from a single-position logit vector."""
    x = logits - logits.max()  # numerical stability
    p = np.exp(x)
    p = p / p.sum()
    p = np.clip(p, 1e-12, 1.0)
    return float(-np.sum(p * np.log(p)))
