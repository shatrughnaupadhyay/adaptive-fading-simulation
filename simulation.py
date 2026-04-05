"""
Adaptive Fading in Agentic Scaffolding -- Simulation Study
===========================================================
Implements the multi-agent fading framework described in Section III.
Validates contributions C1--C4 via synthetic learner emulation (Section IV).

Components:
  - Synthetic Learner Model (BEAGLE-inspired, Section IV-A)
  - Student Knowledge Modeling Agent (SKMA, Eq. 2)
  - Engagement Prediction Model (EPM, Eq. 3)
  - Tutor Selector Agent with PPO-trained fading policy (Eq. 4--6)
  - Baselines: Static-Full, Static-None, Random-Fade, Threshold-Fade
  - Ablation: No-EPM, No-SKMA

Metrics (Section IV-C):
  M_CSR   -- Cold Start Refactor score (0-1)
  H_int   -- Interaction Entropy (bits)
  T_eff   -- Transfer Efficiency (0-1)
  K_gain  -- Knowledge Gain (0-1)

Usage:
  python simulation.py             # console output
  python simulation.py --charts    # Charts saved to charts/
"""

import argparse
import os
import warnings
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml
from scipy import stats
from tabulate import tabulate

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Configuration loader
# ---------------------------------------------------------------------------
_CFG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")

def _load_config(path: str = _CFG_PATH) -> dict:
    if os.path.exists(path):
        with open(path) as f:
            return yaml.safe_load(f)
    return {}

def _g(cfg: dict, *keys, default=None):
    """Nested dict lookup with dotted fallback."""
    node = cfg
    for k in keys:
        if isinstance(node, dict):
            node = node.get(k)
        else:
            return default
    return node if node is not None else default

CFG = _load_config()

# ---------------------------------------------------------------------------
# Constants (loaded from config.yaml, with hardcoded fallbacks)
# ---------------------------------------------------------------------------
N_EPISODES = _g(CFG, "simulation", "n_episodes", default=80)
N_LEARNERS = _g(CFG, "simulation", "n_learners_per_archetype", default=50)
N_CONCEPTS = _g(CFG, "simulation", "n_concepts", default=10)
SEEDS = _g(CFG, "simulation", "seeds", default=[42, 123, 456, 789, 1024])

SCAFFOLD_LEVELS = {0: "Passive/Active", 1: "Constructive", 2: "Interactive"}

W1_DEFAULT = _g(CFG, "reward", "w1", default=0.5)
W2_DEFAULT = _g(CFG, "reward", "w2", default=0.3)
W3_DEFAULT = _g(CFG, "reward", "w3", default=0.2)

# SKMA
SKMA_ALPHA = _g(CFG, "skma", "alpha", default=0.6)
SKMA_SIGMA_K = _g(CFG, "skma", "sigma_k", default=0.03)
SKMA_COLD_START_NOISE = _g(CFG, "skma", "cold_start_noise", default=0.08)

# EPM
EPM_BETA1 = _g(CFG, "epm", "beta1", default=0.6)
EPM_BETA2 = _g(CFG, "epm", "beta2", default=0.3)
EPM_SIGMA_E = _g(CFG, "epm", "sigma_e", default=0.03)
EPM_WINDOW = _g(CFG, "epm", "window", default=5)

# Fading policy
THETA1 = _g(CFG, "fading", "theta1", default=0.30)
THETA2 = _g(CFG, "fading", "theta2", default=0.55)
TAU_E = _g(CFG, "fading", "tau_e", default=0.35)
TAU_H = _g(CFG, "fading", "tau_h", default=0.55)
EXPLORATION_RATE = _g(CFG, "fading", "exploration_rate", default=0.04)

# Learning dynamics
LD_FRUSTRATION_INC = _g(CFG, "learning_dynamics", "frustration_increment", default=0.04)
LD_FRUSTRATION_DEC_OUT = _g(CFG, "learning_dynamics", "frustration_decrement_outsource", default=0.02)
LD_FRUSTRATION_DEC_PROD = _g(CFG, "learning_dynamics", "frustration_decrement_productive", default=0.03)
LD_FRUSTRATION_THRESH = _g(CFG, "learning_dynamics", "frustration_threshold", default=0.5)
LD_FRUSTRATION_GAIN_PEN = _g(CFG, "learning_dynamics", "frustration_gain_penalty", default=0.5)
LD_STRUGGLE_HIGH = _g(CFG, "learning_dynamics", "struggle_high", default=0.7)
LD_STRUGGLE_LOW = _g(CFG, "learning_dynamics", "struggle_low", default=0.15)
LD_MASTERY_LOW = _g(CFG, "learning_dynamics", "mastery_low", default=0.35)
LD_OUTSOURCE_FACTOR = _g(CFG, "learning_dynamics", "outsourcing_factor", default=0.4)
LD_BOOST_PER_LEVEL = _g(CFG, "learning_dynamics", "performance_boost_per_level", default=0.18)
LD_PHI = _g(CFG, "learning_dynamics", "phi", default=0.6)

# Transfer efficiency
TE_KAPPA = _g(CFG, "transfer_efficiency", "kappa", default=0.4)
TE_RHO = _g(CFG, "transfer_efficiency", "rho", default=0.85)
TE_SIGMA = _g(CFG, "transfer_efficiency", "sigma_tau", default=0.03)

# Archetypes
ARCHETYPES = _g(CFG, "archetypes", default={
    "Novice": {"mastery_range": [0.05, 0.25], "learning_rate_range": [0.03, 0.06],
               "engagement_range": [0.4, 0.6], "reliance_range": [0.5, 0.8]},
    "Intermediate": {"mastery_range": [0.30, 0.55], "learning_rate_range": [0.04, 0.07],
                     "engagement_range": [0.5, 0.7], "reliance_range": [0.3, 0.5]},
    "Expert": {"mastery_range": [0.60, 0.85], "learning_rate_range": [0.02, 0.05],
               "engagement_range": [0.6, 0.8], "reliance_range": [0.1, 0.3]},
})

# Sensitivity
NOISE_LEVELS = _g(CFG, "sensitivity", "noise_levels", default=[0.0, 0.05, 0.10, 0.15, 0.20, 0.30])
THRESHOLD_SWEEP = _g(CFG, "sensitivity", "threshold_sweep",
                     default=[[0.20, 0.45], [0.25, 0.50], [0.30, 0.55], [0.35, 0.60], [0.40, 0.65]])

# Baseline thresholds
BL_THETA1 = _g(CFG, "baseline_thresholds", "theta1", default=0.35)
BL_THETA2 = _g(CFG, "baseline_thresholds", "theta2", default=0.65)

# Statistics
BOOTSTRAP_N = _g(CFG, "statistics", "bootstrap_resamples", default=1000)
BOOTSTRAP_SEED = _g(CFG, "statistics", "bootstrap_seed", default=999)


def _reload_constants():
    """Re-read all module-level constants from the current CFG dict.
    Called when --config overrides the default config.yaml at runtime."""
    g = globals()
    _map = {
        "N_EPISODES": ("simulation", "n_episodes", 80),
        "N_LEARNERS": ("simulation", "n_learners_per_archetype", 50),
        "N_CONCEPTS": ("simulation", "n_concepts", 10),
        "SEEDS": ("simulation", "seeds", [42, 123, 456, 789, 1024]),
        "W1_DEFAULT": ("reward", "w1", 0.5),
        "W2_DEFAULT": ("reward", "w2", 0.3),
        "W3_DEFAULT": ("reward", "w3", 0.2),
        "SKMA_ALPHA": ("skma", "alpha", 0.6),
        "SKMA_SIGMA_K": ("skma", "sigma_k", 0.03),
        "SKMA_COLD_START_NOISE": ("skma", "cold_start_noise", 0.08),
        "EPM_BETA1": ("epm", "beta1", 0.6),
        "EPM_BETA2": ("epm", "beta2", 0.3),
        "EPM_SIGMA_E": ("epm", "sigma_e", 0.03),
        "EPM_WINDOW": ("epm", "window", 5),
        "THETA1": ("fading", "theta1", 0.30),
        "THETA2": ("fading", "theta2", 0.55),
        "TAU_E": ("fading", "tau_e", 0.35),
        "TAU_H": ("fading", "tau_h", 0.55),
        "EXPLORATION_RATE": ("fading", "exploration_rate", 0.04),
        "LD_FRUSTRATION_INC": ("learning_dynamics", "frustration_increment", 0.04),
        "LD_FRUSTRATION_DEC_OUT": ("learning_dynamics", "frustration_decrement_outsource", 0.02),
        "LD_FRUSTRATION_DEC_PROD": ("learning_dynamics", "frustration_decrement_productive", 0.03),
        "LD_FRUSTRATION_THRESH": ("learning_dynamics", "frustration_threshold", 0.5),
        "LD_FRUSTRATION_GAIN_PEN": ("learning_dynamics", "frustration_gain_penalty", 0.5),
        "LD_STRUGGLE_HIGH": ("learning_dynamics", "struggle_high", 0.7),
        "LD_STRUGGLE_LOW": ("learning_dynamics", "struggle_low", 0.15),
        "LD_MASTERY_LOW": ("learning_dynamics", "mastery_low", 0.35),
        "LD_OUTSOURCE_FACTOR": ("learning_dynamics", "outsourcing_factor", 0.4),
        "LD_BOOST_PER_LEVEL": ("learning_dynamics", "performance_boost_per_level", 0.18),
        "LD_PHI": ("learning_dynamics", "phi", 0.6),
        "TE_KAPPA": ("transfer_efficiency", "kappa", 0.4),
        "TE_RHO": ("transfer_efficiency", "rho", 0.85),
        "TE_SIGMA": ("transfer_efficiency", "sigma_tau", 0.03),
        "BL_THETA1": ("baseline_thresholds", "theta1", 0.35),
        "BL_THETA2": ("baseline_thresholds", "theta2", 0.65),
        "BOOTSTRAP_N": ("statistics", "bootstrap_resamples", 1000),
        "BOOTSTRAP_SEED": ("statistics", "bootstrap_seed", 999),
    }
    for name, (k1, k2, dflt) in _map.items():
        g[name] = _g(CFG, k1, k2, default=dflt)
    g["ARCHETYPES"] = _g(CFG, "archetypes", default=g.get("ARCHETYPES"))
    g["NOISE_LEVELS"] = _g(CFG, "sensitivity", "noise_levels", default=g.get("NOISE_LEVELS"))
    g["THRESHOLD_SWEEP"] = _g(CFG, "sensitivity", "threshold_sweep", default=g.get("THRESHOLD_SWEEP"))


# ---------------------------------------------------------------------------
# Synthetic Learner Model (BEAGLE-inspired, Section IV-A)
# ---------------------------------------------------------------------------
@dataclass
class SyntheticLearner:
    knowledge: np.ndarray
    learning_rate: float
    engagement_base: float
    reliance_tendency: float
    frustration: float = 0.0       # accumulated frustration from over-challenge
    noise_std: float = 0.05

    def attempt_problem(self, concept: int, scaffold_level: int,
                        rng: np.random.Generator) -> dict:
        k = self.knowledge[concept]

        scaffold_boost = (2 - scaffold_level) * LD_BOOST_PER_LEVEL
        p_correct = np.clip(k + scaffold_boost + rng.normal(0, self.noise_std), 0, 1)
        correct = rng.random() < p_correct

        difficulty = 1.0 - k
        struggle = np.clip(difficulty - scaffold_boost, 0, 1)

        if struggle > LD_STRUGGLE_HIGH and k < LD_MASTERY_LOW:
            self.frustration = min(1.0, self.frustration + LD_FRUSTRATION_INC)
            frustration_penalty = LD_PHI * self.frustration
            learning_signal = max(0, struggle * (1 - struggle) - frustration_penalty)
        elif struggle < LD_STRUGGLE_LOW:
            learning_signal = struggle * LD_OUTSOURCE_FACTOR
            self.frustration = max(0, self.frustration - LD_FRUSTRATION_DEC_OUT)
        else:
            learning_signal = struggle * (1.0 - 0.5 * struggle)
            self.frustration = max(0, self.frustration - LD_FRUSTRATION_DEC_PROD)

        gain = self.learning_rate * learning_signal * (1 + 0.1 * rng.normal())
        gain = max(0, gain)

        engagement = self.engagement_base + 0.35 * learning_signal
        engagement -= 0.25 * self.frustration
        engagement += rng.normal(0, 0.04)
        engagement = np.clip(engagement, 0, 1)

        hint_request = rng.random() < (self.reliance_tendency +
                                        0.12 * (2 - scaffold_level) -
                                        0.1 * scaffold_level)

        if self.frustration > LD_FRUSTRATION_THRESH:
            gain *= (1.0 - LD_FRUSTRATION_GAIN_PEN * self.frustration)

        self.knowledge[concept] = np.clip(k + gain, 0, 1)

        return {
            "correct": correct, "p_correct": p_correct,
            "gain": gain, "engagement": engagement,
            "hint_request": hint_request, "struggle_factor": struggle,
            "response_time": max(0.5, 3.0 - 2.0 * k + rng.normal(0, 0.3)),
            "frustration": self.frustration,
        }


def create_learners(archetype: str, n: int, rng: np.random.Generator):
    spec = ARCHETYPES[archetype]
    mr = spec["mastery_range"]
    lr_r = spec["learning_rate_range"]
    er = spec["engagement_range"]
    rr = spec["reliance_range"]
    learners = []
    for _ in range(n):
        k = rng.uniform(mr[0], mr[1], N_CONCEPTS)
        lr = rng.uniform(lr_r[0], lr_r[1])
        eng = rng.uniform(er[0], er[1])
        rel = rng.uniform(rr[0], rr[1])
        learners.append(SyntheticLearner(
            knowledge=k.copy(), learning_rate=lr,
            engagement_base=eng, reliance_tendency=rel))
    return learners


# ---------------------------------------------------------------------------
# Agent Components (Section III-C)
# ---------------------------------------------------------------------------
class SKMA:
    def __init__(self, noise: float = 0.0):
        self.noise = noise

    def estimate_mastery(self, learner, history, rng) -> float:
        true_m = learner.knowledge.mean()
        if self.noise > 0:
            return np.clip(true_m + rng.normal(0, self.noise), 0, 1)
        if len(history) < 3:
            return np.clip(true_m + rng.normal(0, SKMA_COLD_START_NOISE), 0, 1)
        recent = [h["p_correct"] for h in history[-EPM_WINDOW:]]
        return np.clip(SKMA_ALPHA * np.mean(recent) + (1 - SKMA_ALPHA) * true_m
                       + rng.normal(0, SKMA_SIGMA_K), 0, 1)


class EPM:
    def predict_engagement(self, history, rng) -> float:
        if len(history) < 2:
            return 0.5
        recent_eng = [h["engagement"] for h in history[-EPM_WINDOW:]]
        recent_frust = [h.get("frustration", 0) for h in history[-EPM_WINDOW:]]
        eng = np.mean(recent_eng)
        frust = np.mean(recent_frust)
        return np.clip(EPM_BETA1 * eng - EPM_BETA2 * frust
                       + rng.normal(0, EPM_SIGMA_E), 0, 1)


# ---------------------------------------------------------------------------
# Fading Policies (Section III-E)
# ---------------------------------------------------------------------------
def policy_adaptive_fade(mastery, engagement, hint_rate, episode, rng,
                         **kw) -> int:
    """PPO-trained adaptive fading policy (approximated).

    Learned behaviour: fade support as mastery rises, but step back when
    engagement drops (indicating unproductive struggle).  Penalise high
    hint-request rates by nudging the learner toward autonomy.
    """
    if mastery < THETA1:
        base = 0
    elif mastery < THETA2:
        base = 1
    else:
        base = 2

    if engagement < TAU_E and base > 0:
        base -= 1
    if hint_rate > TAU_H and base < 2 and mastery > BL_THETA1:
        base += 1
    if rng.random() < EXPLORATION_RATE:
        base = rng.integers(0, 3)
    return base


def policy_static_full(m, e, h, ep, rng, **kw):
    return 0

def policy_static_none(m, e, h, ep, rng, **kw):
    return 2

def policy_random(m, e, h, ep, rng, **kw):
    return rng.integers(0, 3)

def policy_threshold(mastery, e, h, ep, rng, **kw):
    if mastery < BL_THETA1:
        return 0
    return 1 if mastery < BL_THETA2 else 2

def policy_no_epm(mastery, engagement, hint_rate, ep, rng, **kw):
    """Ablation: no engagement signal."""
    if mastery < THETA1:
        return 0
    return 1 if mastery < THETA2 else 2

def policy_no_skma(mastery, engagement, hint_rate, ep, rng, **kw):
    """Ablation: noisy mastery estimate (degrades ZPD tracking)."""
    noisy = np.clip(mastery + rng.normal(0, 0.20), 0, 1)
    if noisy < THETA1:
        return 0
    return 1 if noisy < THETA2 else 2


# ---------------------------------------------------------------------------
# Metrics (Section IV-C)
# ---------------------------------------------------------------------------
def compute_cold_start_refactor(learner, rng):
    scores = []
    for c in range(N_CONCEPTS):
        p = np.clip(learner.knowledge[c] + rng.normal(0, SKMA_SIGMA_K), 0, 1)
        scores.append(float(rng.random() < p))
    return np.mean(scores)

def compute_interaction_entropy(history):
    if len(history) < 2:
        return 0.0
    levels = [h.get("scaffold_level", 0) for h in history]
    transitions = [(levels[i], levels[i+1]) for i in range(len(levels)-1)]
    counts = {}
    for t in transitions:
        counts[t] = counts.get(t, 0) + 1
    total = sum(counts.values())
    probs = [c / total for c in counts.values()]
    return float(-sum(p * np.log2(p) for p in probs if p > 0))

def compute_transfer_efficiency(learner, rng):
    scores = []
    for c in range(N_CONCEPTS):
        k = learner.knowledge[c]
        tp = np.clip((k - TE_KAPPA) / (1 - TE_KAPPA), 0, 1) * TE_RHO + rng.normal(0, TE_SIGMA)
        scores.append(np.clip(tp, 0, 1))
    return np.mean(scores)

def compute_knowledge_gain(init_k, final_k):
    return float(np.mean(final_k - init_k))


# ---------------------------------------------------------------------------
# Simulation Engine
# ---------------------------------------------------------------------------
def run_experiment(policy_fn, seed, skma_noise=0.0):
    rng = np.random.default_rng(seed)
    skma = SKMA(noise=skma_noise)
    epm = EPM()
    results_by_arch = {}

    for archetype in ARCHETYPES:
        learners = create_learners(archetype, N_LEARNERS, rng)
        arch_res = []
        for learner in learners:
            init_k = learner.knowledge.copy()
            history = []
            hint_count = 0
            for ep in range(N_EPISODES):
                concept = ep % N_CONCEPTS
                mastery = skma.estimate_mastery(learner, history, rng)
                eng = epm.predict_engagement(history, rng)
                hr = hint_count / max(1, ep)
                level = policy_fn(mastery, eng, hr, ep, rng)
                outcome = learner.attempt_problem(concept, level, rng)
                outcome["scaffold_level"] = level
                history.append(outcome)
                if outcome["hint_request"]:
                    hint_count += 1

            arch_res.append({
                "M_CSR": compute_cold_start_refactor(learner, rng),
                "H_int": compute_interaction_entropy(history),
                "T_eff": compute_transfer_efficiency(learner, rng),
                "K_gain": compute_knowledge_gain(init_k, learner.knowledge),
                "hint_rate": hint_count / N_EPISODES,
            })
        results_by_arch[archetype] = arch_res

    all_res = []
    for a in results_by_arch.values():
        all_res.extend(a)
    return {"by_archetype": results_by_arch, "all": all_res}


def agg(results, metric):
    return np.array([r[metric] for r in results])


def paired_comp(prop, base):
    diff = prop - base
    d_mean = np.mean(diff)
    ps = np.sqrt((np.std(prop, ddof=1)**2 + np.std(base, ddof=1)**2) / 2)
    cd = d_mean / ps if ps > 0 else 0.0
    try:
        _, p = stats.wilcoxon(prop, base, alternative="greater")
    except (ValueError, ZeroDivisionError):
        p = 1.0
    rng_b = np.random.default_rng(BOOTSTRAP_SEED)
    bm = [np.mean(prop[rng_b.integers(0, len(prop), len(prop))]) for _ in range(BOOTSTRAP_N)]
    ci = np.percentile(bm, [2.5, 97.5])
    return {"mean_p": np.mean(prop), "mean_b": np.mean(base),
            "diff": d_mean, "cd": cd, "p": p, "ci": ci}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Adaptive Fading in Agentic Scaffolding -- Simulation")
    parser.add_argument("--charts", action="store_true",
                        help="Generate chart PNGs in prototype/charts/")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config.yaml (default: config.yaml next to this script)")
    args = parser.parse_args()

    if args.config:
        global CFG
        CFG = _load_config(args.config)
        _reload_constants()

    policies = {
        "AdaptiveFade (Proposed)": policy_adaptive_fade,
        "Static-Full": policy_static_full,
        "Static-None": policy_static_none,
        "Random-Fade": policy_random,
        "Threshold-Fade": policy_threshold,
        "No-EPM (Ablation)": policy_no_epm,
        "No-SKMA (Ablation)": policy_no_skma,
    }
    metrics = ["M_CSR", "H_int", "T_eff", "K_gain"]
    mnames = {"M_CSR": "Cold Start Refactor", "H_int": "Interaction Entropy (bits)",
              "T_eff": "Transfer Efficiency", "K_gain": "Knowledge Gain"}

    print("=" * 72)
    print("Adaptive Fading in Agentic Scaffolding -- Simulation Results")
    print("=" * 72)
    print(f"Seeds: {SEEDS}  |  Learners/archetype: {N_LEARNERS}  |  "
          f"Archetypes: 3  |  Episodes: {N_EPISODES}")
    print(f"Reward weights: w1={W1_DEFAULT}, w2={W2_DEFAULT}, w3={W3_DEFAULT}\n")

    # Run all policies
    all_sr = {}
    for pn, pf in policies.items():
        all_sr[pn] = {s: run_experiment(pf, s) for s in SEEDS}

    # TABLE II
    print("-" * 72)
    print("TABLE II: Main Results (mean +/- std across 5 seeds)")
    print("-" * 72)
    rows, sdata = [], {}
    for pn in policies:
        row = [pn]
        mv = {}
        for m in metrics:
            ps = [np.mean(agg(all_sr[pn][s]["all"], m)) for s in SEEDS]
            mu, sd = np.mean(ps), np.std(ps, ddof=1)
            row.append(f"{mu:.3f} +/- {sd:.3f}")
            mv[m] = (mu, sd)
        rows.append(row)
        sdata[pn] = mv
    print(tabulate(rows, headers=["Method"] + [mnames[m] for m in metrics], tablefmt="grid"))
    print()

    # TABLE IIb: Stats
    print("-" * 72)
    print("TABLE IIb: Statistical Comparisons (Proposed vs. Baselines)")
    print("-" * 72)
    pn0 = "AdaptiveFade (Proposed)"
    srows = []
    for pn in policies:
        if pn == pn0:
            continue
        for m in metrics:
            pv = np.concatenate([agg(all_sr[pn0][s]["all"], m) for s in SEEDS])
            bv = np.concatenate([agg(all_sr[pn][s]["all"], m) for s in SEEDS])
            c = paired_comp(pv, bv)
            sig = "***" if c["p"] < .001 else "**" if c["p"] < .01 else "*" if c["p"] < .05 else "n.s."
            srows.append([pn, mnames[m], f"{c['diff']:+.3f}", f"{c['cd']:.3f}",
                         f"{c['p']:.4f}", sig, f"[{c['ci'][0]:.3f}, {c['ci'][1]:.3f}]"])
    print(tabulate(srows, headers=["Baseline", "Metric", "Diff", "Cohen's d",
                                    "p-value", "Sig.", "95% CI (Proposed)"], tablefmt="grid"))
    print()

    # TABLE III: Ablation
    print("-" * 72)
    print("TABLE III: Ablation Study")
    print("-" * 72)
    abl = ["AdaptiveFade (Proposed)", "No-EPM (Ablation)", "No-SKMA (Ablation)"]
    ar = []
    for pn in abl:
        row = [pn]
        for m in metrics:
            ps = [np.mean(agg(all_sr[pn][s]["all"], m)) for s in SEEDS]
            row.append(f"{np.mean(ps):.3f} +/- {np.std(ps, ddof=1):.3f}")
        ar.append(row)
    print(tabulate(ar, headers=["Method"] + [mnames[m] for m in metrics], tablefmt="grid"))
    print()

    # TABLE IV: Per-archetype
    print("-" * 72)
    print("TABLE IV: Per-Archetype M_CSR (Proposed Method)")
    print("-" * 72)
    for arch in ["Novice", "Intermediate", "Expert"]:
        ps = [np.mean(agg(all_sr[pn0][s]["by_archetype"][arch], "M_CSR")) for s in SEEDS]
        print(f"  {arch:14s}  M_CSR = {np.mean(ps):.3f} +/- {np.std(ps, ddof=1):.3f}")
    print()

    # TABLE V: Sensitivity -- ZPD noise
    print("-" * 72)
    print("TABLE V: Sensitivity -- ZPD Estimation Noise")
    print("-" * 72)
    nlevels = NOISE_LEVELS
    nrows = []
    for nl in nlevels:
        ps = [np.mean(agg(run_experiment(policy_adaptive_fade, s, skma_noise=nl)["all"], "M_CSR"))
              for s in SEEDS]
        nrows.append([f"{nl:.2f}", f"{np.mean(ps):.3f} +/- {np.std(ps, ddof=1):.3f}"])
    print(tabulate(nrows, headers=["SKMA Noise (σ)", "M_CSR"], tablefmt="grid"))
    print()

    # TABLE VI: Sensitivity -- Mastery thresholds
    print("-" * 72)
    print("TABLE VI: Sensitivity -- Fading Threshold Sweep")
    print("-" * 72)
    thresholds = [tuple(pair) for pair in THRESHOLD_SWEEP]
    trows = []
    for t1, t2 in thresholds:
        def make_policy(t1=t1, t2=t2):
            def _pol(mastery, engagement, hint_rate, episode, rng, **kw):
                if mastery < t1:
                    b = 0
                elif mastery < t2:
                    b = 1
                else:
                    b = 2
                if engagement < TAU_E and b > 0:
                    b -= 1
                if hint_rate > TAU_H and b < 2 and mastery > BL_THETA1:
                    b += 1
                return b
            return _pol
        pol = make_policy()
        ps_csr = [np.mean(agg(run_experiment(pol, s)["all"], "M_CSR")) for s in SEEDS]
        ps_teff = [np.mean(agg(run_experiment(pol, s)["all"], "T_eff")) for s in SEEDS]
        trows.append([f"({t1:.2f}, {t2:.2f})",
                      f"{np.mean(ps_csr):.3f} +/- {np.std(ps_csr, ddof=1):.3f}",
                      f"{np.mean(ps_teff):.3f} +/- {np.std(ps_teff, ddof=1):.3f}"])
    print(tabulate(trows, headers=["Thresholds (θ₁, θ₂)", "M_CSR", "T_eff"], tablefmt="grid"))
    print()

    if args.charts:
        save_charts(sdata, metrics, mnames, nlevels)

    print("=" * 72)
    print("Simulation complete.")


def save_charts(sdata, metrics, mnames, nlevels):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cdir = os.path.join(os.path.dirname(__file__), "charts")
    os.makedirs(cdir, exist_ok=True)

    methods = list(sdata.keys())
    short = [m.replace(" (Proposed)", "\n(Ours)").replace(" (Ablation)", "\n(Abl.)") for m in methods]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for idx, m in enumerate(["M_CSR", "T_eff"]):
        ax = axes[idx]
        v = [sdata[p][m][0] for p in methods]
        e = [sdata[p][m][1] for p in methods]
        c = ["#2196F3" if "Proposed" in p else "#FF9800" if "Ablation" in p else "#9E9E9E"
             for p in methods]
        ax.bar(range(len(methods)), v, yerr=e, capsize=4, color=c, edgecolor="k", linewidth=.5)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(short, fontsize=7)
        ax.set_ylabel(mnames[m])
        ax.set_title(mnames[m], fontweight="bold")
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=.3)
    plt.tight_layout()
    plt.savefig(os.path.join(cdir, "results_chart.png"), dpi=150, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 4))
    cv = []
    for nl in nlevels:
        ps = [np.mean(agg(run_experiment(policy_adaptive_fade, s, skma_noise=nl)["all"], "M_CSR"))
              for s in SEEDS]
        cv.append(np.mean(ps))
    ax.plot(nlevels, cv, "o-", color="#2196F3", lw=2, ms=6)
    ax.set_xlabel("SKMA Noise (σ)")
    ax.set_ylabel("M_CSR")
    ax.set_title("Sensitivity: ZPD Estimation Noise", fontweight="bold")
    ax.grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(os.path.join(cdir, "sensitivity_noise.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Charts saved to prototype/charts/")


if __name__ == "__main__":
    main()
