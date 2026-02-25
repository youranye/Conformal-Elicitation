"""
Calculate expectations for randomized-tie conformal prediction set size.

Setting:
- X takes values in a finite set (default: {x1, x2})
- Y takes values in a finite label set (default: {1,2})
- beta is the marginal probability P(X=x1) (only used when X_VALUES has 2 entries and USE_EXPLICIT_PX=False)
- We randomly draw a calibration set (size N_CAL) and an independent test point from the same distribution
- Conformity score is the model's predicted probability for the label: alpha(x,y) = p_hat(y | x)
- Randomized tie-breaking uses tau ~ Uniform(0,1) when scores tie

We compute:
- E[|Gamma(X_test)|]      expected prediction set size
- P(Y_test in Gamma)      marginal coverage
- P(|Gamma|=0)            empty-set rate
- P(|Gamma|=K)            full-set rate
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple


# =========================
# 1) VARIABLES
# =========================

LABELS: List[int] = [1, 2]
EPSILON: float = 0.05
N_CAL: int = 1

SHARED_TAU_ACROSS_LABELS: bool = True

X_VALUES: List[str] = ["x1", "x2"]

# beta = P(X=x1). Only used when USE_EXPLICIT_PX=False and len(X_VALUES)==2.
BETA: float = 0.5

# If you want >2 x-values, set USE_EXPLICIT_PX=True and define PX below.
USE_EXPLICIT_PX: bool = False
PX: Dict[str, float] = {
    "x1": 0.5,
    "x2": 0.5,
}

# Predictor probabilities p_hat(y|x). Lists aligned with LABELS; each sums to 1.
PHAT: Dict[str, List[float]] = {
    "x1": [0.5, 0.5],
    "x2": [0.5, 0.5],
}

# Fix test x if desired (None means test x is sampled from PX)
FIXED_TEST_X: Optional[str] = None


# =========================
# 2) UTILS
# =========================

TOL = 1e-12

def _check_prob_vector(p: Sequence[float], tol: float = 1e-9) -> None:
    if any(pi < -tol for pi in p):
        raise ValueError(f"Probability vector has negative entry: {p}")
    s = float(sum(p))
    if abs(s - 1.0) > 1e-6:
        raise ValueError(f"Probability vector must sum to 1. Got sum={s} for p={p}")

def clamp01(x: float) -> float:
    return 0.0 if x <= 0.0 else (1.0 if x >= 1.0 else x)

def count_lt_eq(values: Sequence[float], t: float) -> Tuple[int, int]:
    lt = 0
    eq = 0
    for v in values:
        if v < t - TOL:
            lt += 1
        elif abs(v - t) <= TOL:
            eq += 1
    return lt, eq

def build_px() -> Dict[str, float]:
    if USE_EXPLICIT_PX:
        px = dict(PX)
    else:
        if len(X_VALUES) != 2:
            raise ValueError("If USE_EXPLICIT_PX=False, X_VALUES must have exactly 2 entries.")
        x1, x2 = X_VALUES
        px = {x1: BETA, x2: 1.0 - BETA}
    _check_prob_vector(list(px.values()))
    for x in X_VALUES:
        if x not in px:
            raise ValueError(f"PX missing x={x}")
    return px

def score(x: str, y: int) -> float:
    """Conformity score alpha(x,y) = p_hat(y|x)."""
    probs = PHAT[x]
    _check_prob_vector(probs)
    idx = LABELS.index(y)
    return float(probs[idx])


# =========================
# 3) ENUMERATION OF DRAWS
# =========================

@dataclass(frozen=True)
class Outcome:
    x: str
    y: int
    weight: float  # probability under data distribution

def enumerate_one_draw(px: Dict[str, float]) -> List[Outcome]:
    outs: List[Outcome] = []
    for x in X_VALUES:
        if x not in PHAT:
            raise ValueError(f"PHAT missing x={x}")
        probs = PHAT[x]
        _check_prob_vector(probs)
        for y, py in zip(LABELS, probs):
            outs.append(Outcome(x=x, y=y, weight=px[x] * py))
    return outs


# =========================
# 4) TAU-INTEGRATED STATS
# =========================

def include_prob_indep_tau(
    cal_scores: Sequence[float],
    test_score: float,
    epsilon: float,
    n_cal: int,
) -> float:
    """Independent tau per label: P(include) exactly."""
    lt, eq_cal = count_lt_eq(cal_scores, test_score)
    eq = eq_cal + 1
    denom = n_cal + 1
    if eq == 0:
        return 1.0 if (lt / denom) > epsilon else 0.0
    t = (denom * epsilon - lt) / eq
    return 1.0 - clamp01(t)

def tau_kind_and_threshold(
    cal_scores: Sequence[float],
    test_score: float,
    epsilon: float,
    n_cal: int,
) -> Tuple[str, float]:
    """Shared tau: label is always included, never included, or included iff tau > t."""
    lt, eq_cal = count_lt_eq(cal_scores, test_score)
    eq = eq_cal + 1
    denom = n_cal + 1
    if eq == 0:
        return ("always", 0.0) if (lt / denom) > epsilon else ("never", 0.0)
    t = (denom * epsilon - lt) / eq
    return ("threshold", t)

def shared_tau_expected_stats(
    cal_scores: Sequence[float],
    test_scores_by_label: Sequence[float],
    epsilon: float,
    n_cal: int,
    true_label_index: int,
) -> Tuple[float, float, float, float]:
    """Integrate over one shared tau ~ Unif(0,1). Returns (E[size], P(cover), P(empty), P(full))."""
    K = len(test_scores_by_label)
    always_in = 0
    thresholds: List[float] = []

    for s in test_scores_by_label:
        kind, t = tau_kind_and_threshold(cal_scores, s, epsilon, n_cal)
        if kind == "always":
            always_in += 1
        elif kind == "threshold":
            thresholds.append(t)

    exp_size = always_in + sum(1.0 - clamp01(t) for t in thresholds)

    kind_true, t_true = tau_kind_and_threshold(cal_scores, test_scores_by_label[true_label_index], epsilon, n_cal)
    if kind_true == "always":
        p_cover = 1.0
    elif kind_true == "never":
        p_cover = 0.0
    else:
        p_cover = 1.0 - clamp01(t_true)

    size_at_0 = always_in + sum(1 for t in thresholds if t < 0.0)
    size_at_1 = always_in + sum(1 for t in thresholds if t < 1.0)
    switch_points = sorted(t for t in thresholds if 0.0 < t < 1.0)

    if size_at_0 != 0:
        p_empty = 0.0
    else:
        p_empty = 1.0 if len(switch_points) == 0 else clamp01(switch_points[0])

    if size_at_1 != K:
        p_full = 0.0
    else:
        p_full = 1.0 if len(switch_points) == 0 else (1.0 - clamp01(switch_points[-1]))

    return exp_size, p_cover, p_empty, p_full


# =========================
# 5) EXPECTATIONS
# =========================

@dataclass
class ExactResults:
    exp_set_size: float
    marginal_coverage: float
    empty_rate: float
    full_rate: float

def compute_exact() -> ExactResults:
    px = build_px()
    one_draw = enumerate_one_draw(px)

    test_outcomes = one_draw
    if FIXED_TEST_X is not None:
        filt = [o for o in one_draw if o.x == FIXED_TEST_X]
        z = sum(o.weight for o in filt)
        if z <= 0:
            raise ValueError(f"FIXED_TEST_X={FIXED_TEST_X} has zero probability.")
        test_outcomes = [Outcome(o.x, o.y, o.weight / z) for o in filt]

    # Enumerate all calibration score sequences, aggregated by tuple of scores
    cal_states: List[Tuple[Tuple[float, ...], float]] = [(tuple(), 1.0)]
    for _ in range(N_CAL):
        new: Dict[Tuple[float, ...], float] = {}
        for scores, w in cal_states:
            for o in one_draw:
                s = score(o.x, o.y)
                scores2 = scores + (s,)
                new[scores2] = new.get(scores2, 0.0) + w * o.weight
        cal_states = list(new.items())

    exp_size = 0.0
    exp_cov = 0.0
    exp_empty = 0.0
    exp_full = 0.0

    for cal_scores, w_cal in cal_states:
        cal_scores_list = list(cal_scores)

        for test_o in test_outcomes:
            w = w_cal * test_o.weight

            test_scores_by_label = [score(test_o.x, y) for y in LABELS]
            true_idx = LABELS.index(test_o.y)

            if SHARED_TAU_ACROSS_LABELS:
                e_size, p_cov, p_emp, p_ful = shared_tau_expected_stats(
                    cal_scores=cal_scores_list,
                    test_scores_by_label=test_scores_by_label,
                    epsilon=EPSILON,
                    n_cal=N_CAL,
                    true_label_index=true_idx,
                )
            else:
                p_in = [include_prob_indep_tau(cal_scores_list, s, EPSILON, N_CAL) for s in test_scores_by_label]
                e_size = sum(p_in)
                p_cov = p_in[true_idx]
                p_emp = 1.0
                p_ful = 1.0
                for p in p_in:
                    p_emp *= (1.0 - p)
                    p_ful *= p

            exp_size += w * e_size
            exp_cov += w * p_cov
            exp_empty += w * p_emp
            exp_full += w * p_ful

    # Sanity check
    total_prob = sum(w for _, w in cal_states) * sum(o.weight for o in test_outcomes)
    if abs(total_prob - 1.0) > 1e-8:
        raise RuntimeError(f"Total probability != 1 (got {total_prob}). Check PX/PHAT.")

    return ExactResults(exp_size, exp_cov, exp_empty, exp_full)

def main() -> None:
    res = compute_exact()
    px = build_px()
    print("==== Randomized-tie conformal ====")
    print(f"LABELS: {LABELS}")
    print(f"X_VALUES: {X_VALUES}")
    print(f"epsilon: {EPSILON}")
    print(f"N_CAL: {N_CAL}")
    print(f"shared_tau_across_labels: {SHARED_TAU_ACROSS_LABELS}")
    print("--- Distribution ---")
    print(f"PX: {px}")
    print(f"PHAT (also used as P(Y|X) here): {PHAT}")
    print(f"FIXED_TEST_X: {FIXED_TEST_X}")
    print("--- Exact results ---")
    print(f"E[set size]: {res.exp_set_size:.12f}")
    print(f"Marginal coverage: {res.marginal_coverage:.12f}")
    print(f"P(empty set): {res.empty_rate:.12f}")
    print(f"P(full set): {res.full_rate:.12f}")

if __name__ == "__main__":
    main()
