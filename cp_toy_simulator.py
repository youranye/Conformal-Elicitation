
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cp_toy_simulator: a small conformal prediction simulator.

Pipeline:
1) Generate 2n points x_1, ..., x_{2n}.
2) Calibration split (first n points):
   - y_cal ~ p_true(y | x) from a ground-truth distribution.
   - p_pred(y | x) from a prediction mapping.
   - Compute conformal scores using p_pred and y_cal.
   - Calibrate a threshold tau (THR or APS) from those scores.
3) Prediction split (last n points):
   - p_pred(y | x) from the same prediction mapping.
   - Build conformal sets using p_pred and tau.
   - Coverage is evaluated w.r.t. the argmax label of p_true(y | x),
     i.e., y_true(x) = argmax_y p_true(y | x).
   - We also report average set size.

This file is intended to be easy to modify for ground-truth and prediction mappings.
"""

from __future__ import annotations
import numpy as np
from typing import Callable, Dict, Any, List, Literal, Tuple
from dataclasses import dataclass, asdict
import math
import json
import datetime
import os

def write_log(log_path: str, text: str, print_too: bool = True) -> None:
    """Append text directly to log file (no timestamp)."""
    import os
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(text + "\n" + "\n")
    if print_too:
        print(text)


# ---------------------------
# Config
# ---------------------------

@dataclass
class SimConfig:
    n: int = 20
    K: int = 5
    alpha: float = 0.1
    method: Literal["thr", "aps"] = "aps"
    x_dim: int = 1
    x_low: float = 0.0
    x_high: float = 1.0
    seed: int = 0

def print_hparams(cfg: SimConfig) -> None:
    print("=== Hyperparameters ===")
    print(json.dumps(asdict(cfg), indent=2, ensure_ascii=False))

def set_seed(seed: int) -> None:
    np.random.seed(seed)

# ---------------------------
# Sampling x
# ---------------------------

def sample_x(n: int, dim: int, low: float = 0.0, high: float = 1.0) -> np.ndarray:
    return np.random.uniform(low, high, size=(n, dim))

# ---------------------------
# Generic label sampling
# ---------------------------

def sample_labels(probs: np.ndarray) -> np.ndarray:
    cs = probs.cumsum(axis=1)
    r = np.random.rand(probs.shape[0], 1)
    return (cs < r).sum(axis=1)

# ---------------------------
# Example ground-truth distributions
# ---------------------------

def ptrue_gaussian_1d(X: np.ndarray, K: int, kwargs: Dict[str, Any]) -> np.ndarray:
    """Example ground-truth: K Gaussian bumps over [0,1] in 1D."""
    sharpness = kwargs.get("sharpness", 15.0)
    n, _ = X.shape
    centers = np.linspace(0.1, 0.9, K)
    x1 = X[:, 0]
    z = np.zeros((n, K), dtype=float)
    for k in range(K):
        z[:, k] = np.exp(-sharpness * (x1 - centers[k])**2)
    z_sum = z.sum(axis=1, keepdims=True)
    z_sum = np.where(z_sum <= 0, 1.0, z_sum)
    z = z / z_sum
    return z

def ptrue_bernoulli_1d(X: np.ndarray, K: int, kwargs: Dict[str, Any]) -> np.ndarray:
    """
    Example ground-truth for K=2: Bernoulli with sigmoid in 1D.
    p(y=1|x) = sigmoid(a * (x - c)).
    """
    if K != 2:
        raise ValueError("ptrue_bernoulli_1d requires K=2")
    a = kwargs.get("a", 10.0)
    c = kwargs.get("c", 0.5)
    x = X[:, 0]
    p1 = 1.0 / (1.0 + np.exp(-a * (x - c)))
    p0 = 1.0 - p1
    probs = np.stack([p0, p1], axis=1)
    probs = np.clip(probs, 1e-9, None)
    probs /= probs.sum(axis=1, keepdims=True)
    return probs

# ---------------------------
# Example prediction mappings
# ---------------------------

def ppred_perfect(X: np.ndarray, K: int, kwargs: Dict[str, Any]) -> np.ndarray:
    """
    Perfect predictor: just calls the ground-truth function passed in kwargs["ptrue_fn"].
    """
    ptrue_fn = kwargs["ptrue_fn"]
    ptrue_kwargs = kwargs.get("ptrue_kwargs", {})
    return ptrue_fn(X, K, ptrue_kwargs)

def ppred_noisy_from_true(X: np.ndarray, K: int, kwargs: Dict[str, Any]) -> np.ndarray:
    """
    Noisy predictor: samples a Dirichlet around the ground-truth distribution.
    """
    ptrue_fn = kwargs["ptrue_fn"]
    ptrue_kwargs = kwargs.get("ptrue_kwargs", {})
    conc = kwargs.get("concentration", 6.0)
    p_true = ptrue_fn(X, K, ptrue_kwargs)
    n, _ = p_true.shape
    out = np.zeros_like(p_true)
    for i in range(n):
        alpha = np.clip(p_true[i] * conc, 1e-6, None)
        out[i] = np.random.dirichlet(alpha)
    return out

def ppred_random_dirichlet(X: np.ndarray, K: int, kwargs: Dict[str, Any]) -> np.ndarray:
    """Completely uninformed predictor: Dirichlet(1)."""
    n, _ = X.shape
    out = np.zeros((n, K))
    for i in range(n):
        out[i] = np.random.dirichlet(np.ones(K))
    return out

# ---------------------------
# Conformal: THR and APS
# ---------------------------

def quantile_with_aps_adjustment(vals: np.ndarray, alpha: float) -> float:
    n = len(vals)
    q = (1 - alpha) * (1 + 1.0 / n)
    idx = int(math.ceil(q * n)) - 1
    idx = min(max(idx, 0), n - 1)
    return np.partition(vals, idx)[idx]

def calibrate_thr(probs: np.ndarray, y: np.ndarray, alpha: float) -> float:
    s = 1.0 - probs[np.arange(len(y)), y]
    q = quantile_with_aps_adjustment(s, alpha)
    t = 1.0 - q
    return t

def aps_e_scores(probs: np.ndarray, y: np.ndarray, add_uniform_tiebreak: bool = True) -> np.ndarray:
    order = np.argsort(-probs, axis=1)
    sorted_probs = np.take_along_axis(probs, order, axis=1)
    csum = np.cumsum(sorted_probs, axis=1)
    pos = np.array([int(np.where(order[i] == y[i])[0][0]) for i in range(len(y))])
    greater_sum = np.where(pos > 0, csum[np.arange(len(y)), pos - 1], 0.0)
    p_y = probs[np.arange(len(y)), y]
    U = np.random.rand(len(y)) if add_uniform_tiebreak else np.ones(len(y))
    return greater_sum + U * p_y

def calibrate_aps(probs: np.ndarray, y: np.ndarray, alpha: float, add_uniform_tiebreak: bool = True) -> float:
    E = aps_e_scores(probs, y, add_uniform_tiebreak=add_uniform_tiebreak)
    tau = quantile_with_aps_adjustment(E, alpha)
    return tau

# ---------------------------
# Prediction sets
# ---------------------------

def predset_thr_one(p: np.ndarray, tau: float) -> np.ndarray:
    return np.where(p >= tau)[0]

def predset_aps_one(p: np.ndarray, tau: float, deterministic_U: float = 1.0) -> np.ndarray:
    order = np.argsort(-p)
    sorted_probs = p[order]
    csum = np.cumsum(sorted_probs)
    S: List[int] = []
    for rank, k in enumerate(order):
        greater_sum = csum[rank - 1] if rank > 0 else 0.0
        E_k = greater_sum + deterministic_U * p[k]
        if E_k <= tau:
            S.append(k)
    return np.array(S, dtype=int)

def build_sets_batch(probs: np.ndarray, method: str, tau: float, deterministic_U: float = 1.0) -> List[np.ndarray]:
    sets: List[np.ndarray] = []
    for i in range(probs.shape[0]):
        p = probs[i]
        if method == "thr":
            S = predset_thr_one(p, tau)
        else:
            S = predset_aps_one(p, tau, deterministic_U=deterministic_U)
        sets.append(S)
    return sets

def evaluate_sets(sets: List[np.ndarray], y_true: np.ndarray) -> Tuple[float, float, np.ndarray]:
    n = len(sets)
    sizes = np.array([len(s) for s in sets])
    contains = np.array([y_true[i] in sets[i] for i in range(n)], dtype=float)
    coverage = float(contains.mean()) if n > 0 else float("nan")
    avg_size = float(sizes.mean()) if n > 0 else float("nan")
    max_size = sizes.max() if n > 0 else 0
    hist = np.zeros(max_size + 1, dtype=int)
    for s in sizes:
        hist[s] += 1
    return coverage, avg_size, hist

# ---------------------------
# Experiment state
# ---------------------------

@dataclass
class ExperimentState:
    cfg: SimConfig
    tau: float
    method: str
    x_cal: np.ndarray
    y_cal_true: np.ndarray
    p_pred_cal: np.ndarray
    x_pred: np.ndarray
    p_pred_pred: np.ndarray
    y_pred_true: np.ndarray

# ---------------------------
# Main pipeline
# ---------------------------

def run_experiment(
    cfg: SimConfig,
    ptrue_fn: Callable[[np.ndarray, int, Dict[str, Any]], np.ndarray],
    ppred_fn: Callable[[np.ndarray, int, Dict[str, Any]], np.ndarray],
    ptrue_kwargs: Dict[str, Any] | None = None,
    ppred_kwargs: Dict[str, Any] | None = None,
) -> ExperimentState:
    if ptrue_kwargs is None:
        ptrue_kwargs = {}
    if ppred_kwargs is None:
        ppred_kwargs = {}

    set_seed(cfg.seed)

    # 1) all x
    X = sample_x(2 * cfg.n, cfg.x_dim, cfg.x_low, cfg.x_high)

    # 2) ground truth distribution and argmax labels for all points
    p_true_all = ptrue_fn(X, cfg.K, ptrue_kwargs)
    y_true_argmax_all = np.argmax(p_true_all, axis=1)

    # 3) ground-truth labels on calibration split (for calibration scores)
    y_cal_true = sample_labels(p_true_all[:cfg.n])

    # 4) prediction mapping on all points
    p_pred_all = ppred_fn(X, cfg.K, ppred_kwargs)

    x_cal, x_pred = X[:cfg.n], X[cfg.n:]
    p_pred_cal, p_pred_pred = p_pred_all[:cfg.n], p_pred_all[cfg.n:]
    y_pred_true = y_true_argmax_all[cfg.n:]

    # 5) calibrate tau using p_pred_cal and y_cal_true
    if cfg.method == "thr":
        tau = calibrate_thr(p_pred_cal, y_cal_true, alpha=cfg.alpha)
    elif cfg.method == "aps":
        tau = calibrate_aps(p_pred_cal, y_cal_true, alpha=cfg.alpha, add_uniform_tiebreak=True)
    else:
        raise ValueError("Unknown method")

    return ExperimentState(
        cfg=cfg,
        tau=tau,
        method=cfg.method,
        x_cal=x_cal,
        y_cal_true=y_cal_true,
        p_pred_cal=p_pred_cal,
        x_pred=x_pred,
        p_pred_pred=p_pred_pred,
        y_pred_true=y_pred_true,
    )

# ---------------------------
# Prediction split evaluation
# ---------------------------

def predict_split(exp: ExperimentState, deterministic_U: float = 1.0) -> Dict[str, Any]:
    sets = build_sets_batch(exp.p_pred_pred, exp.method, exp.tau, deterministic_U=deterministic_U)
    coverage, avg_size, hist = evaluate_sets(sets, exp.y_pred_true)
    return {
        "tau": float(exp.tau),
        "method": exp.method,
        "alpha": exp.cfg.alpha,
        "n_pred": exp.x_pred.shape[0],
        "avg_set_size": float(avg_size),
        "coverage_true_label": float(coverage),
        "size_hist": hist.tolist(),
        # "example_sets_first5": [s.tolist() for s in sets[:5]],
    }

# ---------------------------
# Demo
# ---------------------------

def demo():
    cfg = SimConfig(n=20, K=2, alpha=0.1, method="aps", x_dim=1, seed=42)
    log_path = "./cp_log.txt"   # path can be changed as needed

    print_hparams(cfg)
    write_log(log_path, "=== New Experiment ===")
    write_log(log_path, f"Hyperparameters: {json.dumps(asdict(cfg))}")

    # Log distributions being used
    write_log(log_path, "Ground-truth distribution: ptrue_gaussian_1d")
    write_log(log_path, "Prediction mapping: ppred_noisy_from_true")

    exp = run_experiment(
        cfg,
        ptrue_fn=ptrue_bernoulli_1d,
        ppred_fn=ppred_noisy_from_true,
        ptrue_kwargs={"sharpness": 15.0},
        ppred_kwargs={"ptrue_fn": ptrue_bernoulli_1d, "ptrue_kwargs": {"sharpness": 15.0}, "concentration": 6.0},
    )

    print("\n=== Calibration summary ===")
    print(f"method={exp.method}  alpha={cfg.alpha}  tau={exp.tau:.6f}")
    print(f"calibration set size = {cfg.n}")

    write_log(log_path, f"Calibration: method={exp.method}, alpha={cfg.alpha}, tau={exp.tau:.6f}, n_cal={cfg.n}")

    results = predict_split(exp, deterministic_U=1.0)

    print("\n=== Prediction split stats ===")
    print(json.dumps(results, indent=2, ensure_ascii=False))

    write_log(log_path, f"Prediction: avg_set_size={results['avg_set_size']:.4f}")
    write_log(log_path, f"Prediction: coverage_true_label={results['coverage_true_label']:.4f}")
    write_log(log_path, f"Prediction: size_hist={results['size_hist']}")


if __name__ == "__main__":
    demo()
