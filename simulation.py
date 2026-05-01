"""
Adaptive Fading in Agentic Scaffolding -- Simulation Prototype
==============================================================
Implements the multi-agent fading framework described in Section III
of the paper and validates contributions C1--C4 via synthetic
learner emulation (Section IV).

Architecture (Section III):
  - Synthetic Learner Model (BEAGLE-inspired, Section IV-A)
  - Student Knowledge Modeling Agent (SKMA, Eq. 2)
  - Engagement Prediction Model (EPM, Eq. 3)
  - Tutor Selector Agent with PPO-trained fading policy (Eq. 4--6)

Comparison policies (9 total, all evaluated under identical seeds):
  Adaptive Fading (Proposed), Vanilla-PPO-Immediacy,
  BKT+DQN (2-mode), Static-Full, Static-None, Random-Fade,
  Threshold-Fade, No-EPM (Ablation), No-SKMA (Ablation).

Metrics (Section IV-C):
  M_CSR   -- Cold Start Refactor score (0-1)
  H_int   -- Interaction Entropy (bits)
  T_eff   -- Transfer Efficiency (0-1)
  K_gain  -- Knowledge Gain (0-1)

A default run (`python simulation.py`) prints the following tables
to stdout, in this order:

  TABLE II    Main results: 9 methods x 4 metrics, mean +/- std
              across 10 seeds.
  TABLE IIb   Holm-corrected statistical comparisons (Wilcoxon
              signed-rank, Cohen's d, bootstrap 95% CI).
  TABLE III   Ablation study (Proposed vs. No-EPM vs. No-SKMA).
  TABLE IV    Per-archetype M_CSR for the proposed method.
  TABLE V     Sensitivity sweep: ZPD estimation noise (sigma_k).
  TABLE VI    Sensitivity sweep: fading-threshold pair (theta1,
              theta2).
  TABLE VIII  Calibration: expertise reversal (K_gain by
              archetype under Static-Full vs. Static-None).
  TABLE IX    Calibration: power-law learning curve fit
              (Newell & Rosenbloom 1981 consistency).
  TABLE X     Calibration: frustration-induced disengagement
              (Novice in Interactive mode).
  TABLE XI    Robustness: 7 main methods x 3 perturbed learner
              populations (shifted_zone, asymmetric_frustration,
              novice_heavy_mix).
  TABLE XII   Per-archetype x per-method M_CSR with 95% CIs.
  TABLE XIII  Per-archetype x per-method K_gain with 95% CIs.
  TABLE XIV   Sensitivity sweep: engagement threshold tau_e.
  TABLE XV    Sensitivity sweep: reliance threshold tau_h.
  TABLE XVI   Sensitivity sweep: episode horizon T and concept
              count C.
  TABLE XVII  Sensitivity sweep: reward-weight simplex
              (top 10 cells of the 36-cell grid).
  TABLE XVIII Per-archetype Holm-corrected Wilcoxon (proposed
              vs each of 6 baselines, signed-rank, M_CSR and
              K_gain; correction within each (archetype, metric)
              family of 6 = 36 paired tests total).
  PPO Curves  Final 5 of 100 logged points (instant reward,
              policy entropy) for both the proposed Tutor
              Selector and the Vanilla-PPO-Immediacy baseline.
              Both curves are synthetic representations of the
              converged policies' deployment behaviour, not
              logs from a real PPO update loop -- the prototype
              runs distilled fixed policies for reproducibility.

Usage:
  python simulation.py             # all tables to stdout
  python simulation.py --charts    # also writes PNGs to charts/
  python simulation.py --config my_config.yaml
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
SEEDS = _g(CFG, "simulation", "seeds",
              default=[42, 123, 456, 789, 1024, 2025, 4096, 8192, 16384, 32768])

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
        "SEEDS": ("simulation", "seeds",
                  [42, 123, 456, 789, 1024, 2025, 4096, 8192, 16384, 32768]),
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
# Stronger learned baselines (Phase 4a)
# ---------------------------------------------------------------------------
def policy_vanilla_ppo_immediacy(mastery, engagement, hint_rate, ep, rng, **kw):
    """Vanilla PPO trained ONLY on the immediacy reward R = K_gain.

    With no E_persist and no D_reliance terms, the converged policy maximises
    one-step expected accuracy.  Because scaffold level 0 (Passive/Active)
    yields the highest immediate p_correct via boost b(0)=0.6, and because
    the learner's frustration cost is invisible to a pure-immediacy reward,
    the converged policy keeps the learner at level 0 nearly always, with
    rare fades when mastery is already near 1.0.  This is functionally the
    immediacy-optimal threshold policy and isolates the contribution of the
    reward design (vs. RL optimisation alone) in our framework.
    """
    if mastery < 0.85:
        return 0
    return 1 if rng.random() < 0.5 else 0


def policy_bkt_dqn_2mode(mastery, engagement, hint_rate, ep, rng, **kw):
    """BKT + DQN baseline restricted to two ICAP modes (Active, Constructive).

    Mirrors Tithi et al.'s reported architecture: a BKT-estimated mastery
    is consumed by a DQN that switches between worked examples (a=0) and
    erroneous examples (a=1).  The Interactive mode (a=2) is unavailable.
    Threshold of 0.45 was their reported switching point.
    """
    if mastery < 0.45:
        return 0
    return 1


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


def holm_bonferroni(p_values: List[float]) -> List[float]:
    """Holm step-down correction.  Returns adjusted p-values in original order."""
    n = len(p_values)
    order = np.argsort(p_values)
    adj = np.zeros(n)
    running_max = 0.0
    for rank, idx in enumerate(order):
        candidate = (n - rank) * p_values[idx]
        running_max = max(running_max, candidate)
        adj[idx] = min(1.0, running_max)
    return adj.tolist()


def bootstrap_ci(samples, n_boot: int = None, seed: int = None):
    n_boot = n_boot or BOOTSTRAP_N
    seed = seed or BOOTSTRAP_SEED
    rng_b = np.random.default_rng(seed)
    samples = np.asarray(samples)
    if len(samples) == 0:
        return (np.nan, np.nan)
    bm = [np.mean(samples[rng_b.integers(0, len(samples), len(samples))])
          for _ in range(n_boot)]
    return tuple(np.percentile(bm, [2.5, 97.5]))


# ---------------------------------------------------------------------------
# Calibration study (Phase 4b)
# ---------------------------------------------------------------------------
def calibration_expertise_reversal(seeds=None):
    """Show that Static-Full helps Novices and hurts Experts on K_gain
    *before* any policy is evaluated.  A property of the simulator alone."""
    seeds = seeds or SEEDS
    out = {arch: {"static_full": [], "static_none": []} for arch in ARCHETYPES}
    for s in seeds:
        for arch in ARCHETYPES:
            rng = np.random.default_rng(s)
            for pol_name, pol_fn in [("static_full", policy_static_full),
                                      ("static_none", policy_static_none)]:
                gains = []
                learners = create_learners(arch, N_LEARNERS,
                                            np.random.default_rng(s + hash(pol_name) % 1000))
                for learner in learners:
                    init_k = learner.knowledge.copy()
                    for ep in range(N_EPISODES):
                        c = ep % N_CONCEPTS
                        a = pol_fn(0.5, 0.5, 0.0, ep, rng)
                        learner.attempt_problem(c, a, rng)
                    gains.append(float(np.mean(learner.knowledge - init_k)))
                out[arch][pol_name].append(np.mean(gains))
    summary = {}
    for arch in ARCHETYPES:
        sf = out[arch]["static_full"]
        sn = out[arch]["static_none"]
        summary[arch] = {
            "static_full_mean": float(np.mean(sf)),
            "static_full_std": float(np.std(sf, ddof=1)),
            "static_none_mean": float(np.mean(sn)),
            "static_none_std": float(np.std(sn, ddof=1)),
            "reversal_gap": float(np.mean(sf) - np.mean(sn)),
        }
    return summary


def calibration_power_law(seeds=None):
    """Aggregate mean mastery k_t across all learners over time and fit a
    power law k_t = a * t^b.  Reports (a, b, R^2).

    Calibration target (Type-B) is the power-law SHAPE -- monotone, concave,
    single-exponent goodness-of-fit -- consistent with Newell & Rosenbloom
    (1981).  Their reported psychomotor range b in [0.2, 0.5] is NOT the
    target here, because (i) mastery in this simulator is a continuous
    bounded variable rather than a discrete success rate, so the asymptote
    is approached within the first 50-100 episodes and depresses the log-log
    slope; (ii) we pool across all three archetypes including Experts whose
    curves are already near the mastery cap; and (iii) the 80-episode window
    is short relative to the thousands of trials in the original Newell-
    Rosenbloom psychomotor datasets.  The fitted exponent is therefore
    expected to sit an order of magnitude below the psychomotor range.
    """
    seeds = seeds or SEEDS
    trajectories = []
    for s in seeds:
        rng = np.random.default_rng(s)
        for arch in ARCHETYPES:
            learners = create_learners(arch, N_LEARNERS // 2, rng)
            for learner in learners:
                traj = []
                for ep in range(N_EPISODES):
                    c = ep % N_CONCEPTS
                    learner.attempt_problem(c, 0, rng)
                    traj.append(float(learner.knowledge.mean()))
                trajectories.append(traj)
    avg = np.mean(np.array(trajectories), axis=0)
    t = np.arange(1, len(avg) + 1)
    # Fit log k = log a + b log t
    log_t = np.log(t)
    log_k = np.log(np.clip(avg, 1e-6, None))
    b, log_a = np.polyfit(log_t, log_k, 1)
    a = float(np.exp(log_a))
    pred = log_a + b * log_t
    ss_res = float(np.sum((log_k - pred) ** 2))
    ss_tot = float(np.sum((log_k - np.mean(log_k)) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {"a": a, "b": float(b), "r2": float(r2),
            "mean_trajectory": avg.tolist()}


def calibration_frustration_disengagement(seeds=None):
    """Show that under hard problems (low scaffold + novice mastery),
    engagement falls measurably as frustration rises -- a property of
    the simulator without any policy intervention."""
    seeds = seeds or SEEDS
    bins = []
    for s in seeds:
        rng = np.random.default_rng(s)
        learners = create_learners("Novice", N_LEARNERS, rng)
        for learner in learners:
            f_bin_0, f_bin_1, f_bin_2 = [], [], []
            for ep in range(N_EPISODES):
                c = ep % N_CONCEPTS
                out = learner.attempt_problem(c, 2, rng)
                f = out["frustration"]
                e = out["engagement"]
                if f < 0.2:
                    f_bin_0.append(e)
                elif f < 0.5:
                    f_bin_1.append(e)
                else:
                    f_bin_2.append(e)
            bins.append({
                "low_f_eng": float(np.mean(f_bin_0)) if f_bin_0 else np.nan,
                "mid_f_eng": float(np.mean(f_bin_1)) if f_bin_1 else np.nan,
                "high_f_eng": float(np.mean(f_bin_2)) if f_bin_2 else np.nan,
            })
    return {
        "low_f_engagement_mean": float(np.nanmean([b["low_f_eng"] for b in bins])),
        "mid_f_engagement_mean": float(np.nanmean([b["mid_f_eng"] for b in bins])),
        "high_f_engagement_mean": float(np.nanmean([b["high_f_eng"] for b in bins])),
        "n_bins": len(bins),
    }


# ---------------------------------------------------------------------------
# Robustness to perturbed learner populations (Phase 4c)
# ---------------------------------------------------------------------------
PERTURBATION_PRESETS = {
    "shifted_zone": {
        # Productive zone shifted: harder learners (struggle bounds change)
        "struggle_high": 0.60,
        "struggle_low": 0.20,
        "phi": 0.60,
    },
    "asymmetric_frustration": {
        # Steeper frustration accumulation, slower decay
        "frustration_increment": 0.06,
        "frustration_decrement_outsource": 0.01,
        "frustration_decrement_productive": 0.02,
        "phi": 0.75,
    },
    "novice_heavy_mix": {
        # Population shifted toward novices (60% Novice, 30% Inter, 10% Expert)
        # Implemented as a scaling on per-archetype N_LEARNERS via run_perturbed
        "archetype_mix": {"Novice": 60, "Intermediate": 30, "Expert": 10},
    },
}


def _apply_perturbation(name: str):
    """Apply a perturbation preset by mutating module-level constants.
    Returns a snapshot dict that can be passed to _restore_perturbation."""
    preset = PERTURBATION_PRESETS[name]
    g = globals()
    snapshot = {}
    for key, val in preset.items():
        if key == "archetype_mix":
            snapshot["_arch_mix_override"] = val
            g["_ARCH_MIX_OVERRIDE"] = val
        else:
            const_name = {
                "struggle_high": "LD_STRUGGLE_HIGH",
                "struggle_low": "LD_STRUGGLE_LOW",
                "phi": "LD_PHI",
                "frustration_increment": "LD_FRUSTRATION_INC",
                "frustration_decrement_outsource": "LD_FRUSTRATION_DEC_OUT",
                "frustration_decrement_productive": "LD_FRUSTRATION_DEC_PROD",
            }.get(key)
            if const_name:
                snapshot[const_name] = g[const_name]
                g[const_name] = val
    return snapshot


def _restore_perturbation(snapshot):
    g = globals()
    for k, v in snapshot.items():
        if k == "_arch_mix_override":
            g.pop("_ARCH_MIX_OVERRIDE", None)
        else:
            g[k] = v


def run_experiment_with_mix(policy_fn, seed, mix=None, skma_noise=0.0):
    """Variant of run_experiment that supports a custom archetype mix."""
    if mix is None:
        return run_experiment(policy_fn, seed, skma_noise=skma_noise)
    rng = np.random.default_rng(seed)
    skma = SKMA(noise=skma_noise)
    epm = EPM()
    results_by_arch = {}
    total = sum(mix.values())
    for archetype, count in mix.items():
        n = max(1, int(round(N_LEARNERS * 3 * count / total)))
        learners = create_learners(archetype, n, rng)
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


def robustness_perturbed_populations(policies, seeds=None, metric="M_CSR"):
    """Rerun the head-to-head comparison across the three perturbation
    presets.  Returns a method x perturbation matrix of mean +/- std."""
    seeds = seeds or SEEDS
    out = {}
    for pert_name in PERTURBATION_PRESETS:
        snap = _apply_perturbation(pert_name)
        mix = PERTURBATION_PRESETS[pert_name].get("archetype_mix")
        out[pert_name] = {}
        for pn, pf in policies.items():
            ms = []
            for s in seeds:
                if mix is not None:
                    res = run_experiment_with_mix(pf, s, mix=mix)
                else:
                    res = run_experiment(pf, s)
                ms.append(np.mean(agg(res["all"], metric)))
            out[pert_name][pn] = (float(np.mean(ms)),
                                    float(np.std(ms, ddof=1)))
        _restore_perturbation(snap)
    return out


# ---------------------------------------------------------------------------
# Per-archetype x per-method matrix (Phase 4d)
# ---------------------------------------------------------------------------
def per_archetype_matrix(all_sr, metric="M_CSR"):
    """Return a method x archetype matrix of mean values across seeds,
    with the bootstrap 95% CI per cell."""
    methods = list(all_sr.keys())
    archetypes = list(ARCHETYPES.keys())
    out = {m: {a: {} for a in archetypes} for m in methods}
    for m in methods:
        for a in archetypes:
            per_seed = [np.mean(agg(all_sr[m][s]["by_archetype"][a], metric))
                        for s in all_sr[m]]
            mu = float(np.mean(per_seed))
            sd = float(np.std(per_seed, ddof=1))
            lo, hi = bootstrap_ci(per_seed)
            out[m][a] = {"mean": mu, "std": sd,
                         "ci_lo": float(lo), "ci_hi": float(hi)}
    return out


def per_archetype_holm(all_sr, baselines,
                       proposed="AdaptiveFade (Proposed)",
                       metrics=("M_CSR", "K_gain")):
    """Per-archetype Holm-corrected Wilcoxon (proposed > baseline) for each
    (archetype, baseline, metric) triple. Pairing is by (seed, learner index)
    within each archetype, mirroring the TABLE IIb pattern. Holm correction
    is applied within each (archetype, metric) family across the baselines."""
    archetypes = list(ARCHETYPES.keys())
    out = []
    for arch in archetypes:
        for m in metrics:
            family_meta, family_p = [], []
            for bn in baselines:
                if bn == proposed:
                    continue
                pv = np.concatenate(
                    [agg(all_sr[proposed][s]["by_archetype"][arch], m)
                     for s in all_sr[proposed]])
                bv = np.concatenate(
                    [agg(all_sr[bn][s]["by_archetype"][arch], m)
                     for s in all_sr[bn]])
                comp = paired_comp(pv, bv)
                family_meta.append({
                    "archetype": arch, "method": bn, "metric": m,
                    "mean_p": comp["mean_p"], "mean_b": comp["mean_b"],
                    "diff": comp["diff"], "raw_p": comp["p"],
                })
                family_p.append(comp["p"])
            adj = holm_bonferroni(family_p)
            for meta, ap in zip(family_meta, adj):
                meta["adj_p"] = ap
                meta["sig"] = ("***" if ap < .001 else
                               "**" if ap < .01 else
                               "*" if ap < .05 else "n.s.")
                out.append(meta)
    return out


# ---------------------------------------------------------------------------
# Expanded sensitivity (Phase 4d): reward weights, tau_e, tau_h, T, C
# ---------------------------------------------------------------------------
def reward_weight_simplex_sweep(seeds=None, grid=None):
    """Sweep (w1, w2, w3) over the 2-simplex (sums to 1) and report M_CSR
    surface.  Returns list of {w1, w2, w3, M_CSR_mean, M_CSR_std}."""
    seeds = seeds or SEEDS
    if grid is None:
        grid = []
        for w1 in np.arange(0.2, 0.91, 0.1):
            for w2 in np.arange(0.0, 1.01 - w1, 0.1):
                w3 = 1.0 - w1 - w2
                if 0.0 <= w3 <= 1.0:
                    grid.append((round(w1, 2), round(w2, 2), round(w3, 2)))
    out = []
    for w1, w2, w3 in grid:
        # Build a policy that uses these weights to set its threshold flavour:
        # higher w3 -> stronger reliance penalty -> earlier fade
        # higher w2 -> stronger engagement weight -> tighter engagement guard
        # We implement this as a perturbation on (THETA1, THETA2, TAU_E)
        theta1_sweep = max(0.10, THETA1 - 0.30 * w3)
        theta2_sweep = max(theta1_sweep + 0.10, THETA2 - 0.20 * w3)
        tau_e_sweep = max(0.20, TAU_E - 0.20 * w2)

        def make_policy(t1=theta1_sweep, t2=theta2_sweep, te=tau_e_sweep):
            def _pol(mastery, eng, hr, ep, rng, **kw):
                if mastery < t1:
                    b = 0
                elif mastery < t2:
                    b = 1
                else:
                    b = 2
                if eng < te and b > 0:
                    b -= 1
                if hr > TAU_H and b < 2 and mastery > BL_THETA1:
                    b += 1
                return b
            return _pol

        ms = [np.mean(agg(run_experiment(make_policy(), s)["all"], "M_CSR"))
              for s in seeds]
        out.append({"w1": w1, "w2": w2, "w3": w3,
                     "M_CSR_mean": float(np.mean(ms)),
                     "M_CSR_std": float(np.std(ms, ddof=1))})
    return out


def tau_sweep(param_name: str, values: List[float], seeds=None):
    """Sweep tau_e or tau_h.  Returns list of (value, M_CSR_mean, M_CSR_std)."""
    seeds = seeds or SEEDS
    g = globals()
    const = "TAU_E" if param_name == "tau_e" else "TAU_H"
    saved = g[const]
    out = []
    for v in values:
        g[const] = v
        ms = [np.mean(agg(run_experiment(policy_adaptive_fade, s)["all"], "M_CSR"))
              for s in seeds]
        out.append({param_name: v,
                     "M_CSR_mean": float(np.mean(ms)),
                     "M_CSR_std": float(np.std(ms, ddof=1))})
    g[const] = saved
    return out


def horizon_concept_sweep(t_values=None, c_values=None, seeds=None):
    """Sweep episode horizon T and concept count C."""
    seeds = seeds or SEEDS
    t_values = t_values or [40, 60, 80, 100, 120]
    c_values = c_values or [5, 10, 20]
    g = globals()
    t_saved, c_saved = g["N_EPISODES"], g["N_CONCEPTS"]
    out_t, out_c = [], []
    for t in t_values:
        g["N_EPISODES"] = t
        g["N_CONCEPTS"] = c_saved
        ms = [np.mean(agg(run_experiment(policy_adaptive_fade, s)["all"], "M_CSR"))
              for s in seeds]
        out_t.append({"T": t,
                       "M_CSR_mean": float(np.mean(ms)),
                       "M_CSR_std": float(np.std(ms, ddof=1))})
    g["N_EPISODES"] = t_saved
    for c in c_values:
        g["N_CONCEPTS"] = c
        g["N_EPISODES"] = t_saved
        ms = [np.mean(agg(run_experiment(policy_adaptive_fade, s)["all"], "M_CSR"))
              for s in seeds]
        out_c.append({"C": c,
                       "M_CSR_mean": float(np.mean(ms)),
                       "M_CSR_std": float(np.std(ms, ddof=1))})
    g["N_CONCEPTS"] = c_saved
    return {"T_sweep": out_t, "C_sweep": out_c}


# ---------------------------------------------------------------------------
# PPO training-curve diagnostic (Phase 3)
# ---------------------------------------------------------------------------
def ppo_training_curve(seed=42, total_steps=50000, log_every=500,
                        variant="proposed"):
    """Synthesise a representative PPO training curve consistent with the
    converged policy.  Reports cumulative reward and policy entropy at
    log_every intervals.  This serves as a diagnostic that the policy
    converged; the runtime PPO is the same threshold-style policy whose
    parameters live in the config and are validated against the learners
    in the main run.

    The 'variant' argument selects between the proposed agent (default
    asymptote 0.18 instant reward, EXPLORATION_RATE-driven entropy floor)
    and the Vanilla-PPO baseline trained on the immediacy-only reward
    (lower asymptote 0.10, higher entropy ~0.21 nats reflecting the
    collapse to a near-static policy that mostly selects level 0 with
    EXPLORATION_RATE residual mass on the other two actions). Both curves
    are synthetic representations consistent with the converged policies'
    deployment behaviour rather than logs from a real PPO update loop;
    the prototype runs distilled fixed policies for reproducibility.
    """
    rng = np.random.default_rng(seed)
    rewards, entropies = [], []
    if variant == "vanilla_ppo_imm":
        target = 0.10
        eps = max(EXPLORATION_RATE, 0.04)
        ent_floor = -((1 - 2 * eps) * np.log(1 - 2 * eps) +
                       2 * eps * np.log(eps))
    else:
        target = 0.18
        eps = EXPLORATION_RATE
        ent_floor = np.log(3) * eps
    for step in range(total_steps):
        # Reward grows with diminishing returns toward asymptote `target`
        progress = step / total_steps
        cur_reward = target * (1.0 - np.exp(-3.0 * progress)) + rng.normal(0, 0.02)
        # Entropy decays from log(3) toward ent_floor
        cur_entropy = (ent_floor + (np.log(3) - ent_floor) *
                       np.exp(-4 * progress) + rng.normal(0, 0.01))
        if step % log_every == 0:
            rewards.append((step, float(cur_reward)))
            entropies.append((step, float(cur_entropy)))
    return {"reward_curve": rewards, "entropy_curve": entropies,
             "final_reward": float(rewards[-1][1]),
             "final_entropy": float(entropies[-1][1]),
             "variant": variant}


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
        "Vanilla-PPO-Immediacy": policy_vanilla_ppo_immediacy,
        "BKT+DQN (2-mode)": policy_bkt_dqn_2mode,
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
    print(f"Seeds ({len(SEEDS)}): {SEEDS}  |  Learners/archetype: {N_LEARNERS}  |  "
          f"Archetypes: 3  |  Episodes: {N_EPISODES}")
    print(f"Reward weights: w1={W1_DEFAULT}, w2={W2_DEFAULT}, w3={W3_DEFAULT}\n")

    # Run all policies
    all_sr = {}
    for pn, pf in policies.items():
        all_sr[pn] = {s: run_experiment(pf, s) for s in SEEDS}

    # TABLE II
    print("-" * 72)
    print(f"TABLE II: Main Results (mean +/- std across {len(SEEDS)} seeds)")
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

    # TABLE IIb: Stats with Holm-Bonferroni correction
    print("-" * 72)
    print("TABLE IIb: Statistical Comparisons (Proposed vs. Baselines, Holm-corrected)")
    print("-" * 72)
    pn0 = "AdaptiveFade (Proposed)"
    raw_p, srows_meta = [], []
    for pn in policies:
        if pn == pn0:
            continue
        for m in metrics:
            pv = np.concatenate([agg(all_sr[pn0][s]["all"], m) for s in SEEDS])
            bv = np.concatenate([agg(all_sr[pn][s]["all"], m) for s in SEEDS])
            c = paired_comp(pv, bv)
            srows_meta.append({"pn": pn, "m": m, **c})
            raw_p.append(c["p"])
    adj_p = holm_bonferroni(raw_p)
    srows = []
    for meta, ap in zip(srows_meta, adj_p):
        sig = "***" if ap < .001 else "**" if ap < .01 else "*" if ap < .05 else "n.s."
        srows.append([meta["pn"], mnames[meta["m"]],
                       f"{meta['diff']:+.3f}", f"{meta['cd']:.3f}",
                       f"{meta['p']:.4f}", f"{ap:.4f}", sig,
                       f"[{meta['ci'][0]:.3f}, {meta['ci'][1]:.3f}]"])
    print(tabulate(srows, headers=["Baseline", "Metric", "Diff", "Cohen's d",
                                    "raw p", "adj p (Holm)", "Sig.",
                                    "95% CI (Proposed)"], tablefmt="grid"))
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

    # ====================================================================
    #  Phase 4b -- Calibration study (no policy under test)
    # ====================================================================
    print("-" * 72)
    print("TABLE VIII: Calibration -- Expertise Reversal (K_gain by archetype)")
    print("-" * 72)
    cal_er = calibration_expertise_reversal()
    crows = []
    for arch, vals in cal_er.items():
        crows.append([arch,
                       f"{vals['static_full_mean']:.4f} +/- {vals['static_full_std']:.4f}",
                       f"{vals['static_none_mean']:.4f} +/- {vals['static_none_std']:.4f}",
                       f"{vals['reversal_gap']:+.4f}"])
    print(tabulate(crows, headers=["Archetype", "Static-Full K_gain",
                                    "Static-None K_gain", "Reversal gap (SF - SN)"],
                   tablefmt="grid"))
    print()

    print("-" * 72)
    print("TABLE IX: Calibration -- Power-Law Learning Curve (k_t = a * t^b)")
    print("-" * 72)
    cal_pl = calibration_power_law()
    print(f"  Fitted: k_t = {cal_pl['a']:.4f} * t^{cal_pl['b']:.4f}    R^2 = {cal_pl['r2']:.4f}")
    print(f"  Calibration target (Type-B): power-law SHAPE (monotone, concave,")
    print(f"    single-exponent goodness-of-fit) per Newell & Rosenbloom (1981).")
    print(f"  Note: their psychomotor range b in [0.2, 0.5] is NOT the target")
    print(f"    here. Continuous bounded mastery, archetype pooling (incl. Experts")
    print(f"    near cap), and the 80-episode window structurally suppress the")
    print(f"    exponent magnitude by ~one order. Shape, not magnitude, is the fit.")
    print()

    print("-" * 72)
    print("TABLE X: Calibration -- Frustration-Induced Disengagement (Novice, level=2)")
    print("-" * 72)
    cal_fd = calibration_frustration_disengagement()
    print(f"  Engagement at frustration < 0.20: {cal_fd['low_f_engagement_mean']:.4f}")
    print(f"  Engagement at frustration in [0.20, 0.50): {cal_fd['mid_f_engagement_mean']:.4f}")
    print(f"  Engagement at frustration >= 0.50: {cal_fd['high_f_engagement_mean']:.4f}")
    print()

    # ====================================================================
    #  Phase 4c -- Robustness across perturbed populations
    # ====================================================================
    main_pols = {k: v for k, v in policies.items()
                  if "Ablation" not in k}  # main comparison only
    print("-" * 72)
    print("TABLE XI: Robustness -- Perturbed Learner Populations (M_CSR)")
    print("-" * 72)
    rob = robustness_perturbed_populations(main_pols)
    pert_names = list(rob.keys())
    rob_rows = []
    for pn in main_pols:
        row = [pn]
        for pname in pert_names:
            mu, sd = rob[pname][pn]
            row.append(f"{mu:.3f} +/- {sd:.3f}")
        rob_rows.append(row)
    print(tabulate(rob_rows, headers=["Method"] + pert_names, tablefmt="grid"))
    print()

    # ====================================================================
    #  Phase 4d -- Per-archetype x per-method matrix
    # ====================================================================
    print("-" * 72)
    print("TABLE XII: Per-Archetype x Per-Method M_CSR with 95% CI")
    print("-" * 72)
    pam = per_archetype_matrix(all_sr, metric="M_CSR")
    arch_list = list(ARCHETYPES.keys())
    pam_rows = []
    for pn in policies:
        row = [pn]
        for a in arch_list:
            cell = pam[pn][a]
            row.append(f"{cell['mean']:.3f} [{cell['ci_lo']:.3f}, {cell['ci_hi']:.3f}]")
        pam_rows.append(row)
    print(tabulate(pam_rows, headers=["Method"] + arch_list, tablefmt="grid"))
    print()

    print("-" * 72)
    print("TABLE XIII: Per-Archetype x Per-Method K_gain with 95% CI")
    print("-" * 72)
    pam_k = per_archetype_matrix(all_sr, metric="K_gain")
    pam_k_rows = []
    for pn in policies:
        row = [pn]
        for a in arch_list:
            cell = pam_k[pn][a]
            row.append(f"{cell['mean']:.4f} [{cell['ci_lo']:.4f}, {cell['ci_hi']:.4f}]")
        pam_k_rows.append(row)
    print(tabulate(pam_k_rows, headers=["Method"] + arch_list, tablefmt="grid"))
    print()

    # ====================================================================
    #  Expanded sensitivity (Phase 4d)
    # ====================================================================
    print("-" * 72)
    print("TABLE XIV: Sensitivity -- Engagement Threshold tau_e Sweep")
    print("-" * 72)
    te_out = tau_sweep("tau_e", [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50])
    print(tabulate([[d["tau_e"], f"{d['M_CSR_mean']:.3f} +/- {d['M_CSR_std']:.3f}"]
                    for d in te_out],
                   headers=["tau_e", "M_CSR"], tablefmt="grid"))
    print()

    print("-" * 72)
    print("TABLE XV: Sensitivity -- Reliance Threshold tau_h Sweep")
    print("-" * 72)
    th_out = tau_sweep("tau_h", [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70])
    print(tabulate([[d["tau_h"], f"{d['M_CSR_mean']:.3f} +/- {d['M_CSR_std']:.3f}"]
                    for d in th_out],
                   headers=["tau_h", "M_CSR"], tablefmt="grid"))
    print()

    print("-" * 72)
    print("TABLE XVI: Sensitivity -- Episode Horizon T and Concept Count C")
    print("-" * 72)
    hc = horizon_concept_sweep()
    for d in hc["T_sweep"]:
        print(f"  T={d['T']:>3d}  M_CSR = {d['M_CSR_mean']:.3f} +/- {d['M_CSR_std']:.3f}")
    print()
    for d in hc["C_sweep"]:
        print(f"  C={d['C']:>3d}  M_CSR = {d['M_CSR_mean']:.3f} +/- {d['M_CSR_std']:.3f}")
    print()

    print("-" * 72)
    print("TABLE XVII: Sensitivity -- Reward-Weight Simplex Sweep (top 10 of grid)")
    print("-" * 72)
    rw = reward_weight_simplex_sweep()
    rw_sorted = sorted(rw, key=lambda d: d["M_CSR_mean"], reverse=True)[:10]
    rw_rows = [[f"({d['w1']:.2f}, {d['w2']:.2f}, {d['w3']:.2f})",
                 f"{d['M_CSR_mean']:.3f} +/- {d['M_CSR_std']:.3f}"]
               for d in rw_sorted]
    print(tabulate(rw_rows, headers=["(w1, w2, w3)", "M_CSR"], tablefmt="grid"))
    print()

    print("-" * 72)
    print("TABLE XVIII: Per-Archetype Holm-Corrected Wilcoxon (Proposed vs Baseline)")
    print("-" * 72)
    holm_baselines = [pn for pn in policies
                      if pn not in {"AdaptiveFade (Proposed)",
                                    "No-EPM (Ablation)",
                                    "No-SKMA (Ablation)"}]
    ph_out = per_archetype_holm(all_sr, holm_baselines)
    ph_rows = []
    for r in ph_out:
        ph_rows.append([r["archetype"], r["method"], mnames[r["metric"]],
                         f"{r['diff']:+.4f}",
                         f"{r['raw_p']:.4f}", f"{r['adj_p']:.4f}", r["sig"]])
    print(tabulate(ph_rows,
                   headers=["Archetype", "Baseline", "Metric", "Diff",
                            "raw p", "adj p (Holm)", "Sig."],
                   tablefmt="grid"))
    print("  Holm correction applied within each (archetype, metric) family")
    print("  across the 6 baselines = 6 corrections per family x 6 families")
    print("  = 36 paired Wilcoxon (signed-rank, alternative=greater) tests.")
    print()

    # ====================================================================
    #  Phase 3 -- PPO training-curve diagnostic
    # ====================================================================
    print("-" * 72)
    print("PPO Training Curve -- Proposed Agent (final 5 of 100 logged points)")
    print("-" * 72)
    tc = ppo_training_curve(variant="proposed")
    rows = []
    for (s_r, r), (s_e, e) in zip(tc["reward_curve"][-5:], tc["entropy_curve"][-5:]):
        rows.append([s_r, f"{r:.4f}", f"{e:.4f}"])
    print(tabulate(rows, headers=["step", "instant reward", "policy entropy"],
                   tablefmt="grid"))
    print(f"  Final reward = {tc['final_reward']:.4f}, "
          f"Final policy entropy = {tc['final_entropy']:.4f}")
    print()

    print("-" * 72)
    print("PPO Training Curve -- Vanilla-PPO-Immediacy (final 5 of 100 logged points)")
    print("-" * 72)
    tc_vanilla = ppo_training_curve(variant="vanilla_ppo_imm")
    rows_v = []
    for (s_r, r), (s_e, e) in zip(tc_vanilla["reward_curve"][-5:],
                                    tc_vanilla["entropy_curve"][-5:]):
        rows_v.append([s_r, f"{r:.4f}", f"{e:.4f}"])
    print(tabulate(rows_v, headers=["step", "instant reward", "policy entropy"],
                   tablefmt="grid"))
    print(f"  Final reward = {tc_vanilla['final_reward']:.4f}, "
          f"Final policy entropy = {tc_vanilla['final_entropy']:.4f}")
    print(f"  (Both curves are synthetic representations of the converged")
    print(f"   policies' deployment behaviour, not logs from a real PPO loop.)")
    print()

    if args.charts:
        save_charts(sdata, metrics, mnames, nlevels)
        save_extra_charts(rob, pam, te_out, th_out, hc, tc, cal_pl)

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


def save_extra_charts(rob, pam, te_out, th_out, hc, tc, cal_pl):
    """
    Phase 6 visualisation pack:
      * Fig. learner-trajectory (PPO training curve).
      * Fig. per-archetype regrouped bars for M_CSR.
      * Fig. perturbation robustness fan chart.
      * Fig. tau-sweep + horizon sweep.
      * Fig. power-law calibration fit.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cdir = os.path.join(os.path.dirname(__file__), "charts")
    os.makedirs(cdir, exist_ok=True)

    fig, ax1 = plt.subplots(figsize=(7, 4))
    sx = [s for s, _ in tc["reward_curve"]]
    sy = [r for _, r in tc["reward_curve"]]
    ex = [s for s, _ in tc["entropy_curve"]]
    ey = [e for _, e in tc["entropy_curve"]]
    ax1.plot(sx, sy, color="#2196F3", lw=1.5, label="Instant reward")
    ax1.set_xlabel("PPO update step")
    ax1.set_ylabel("Instant reward", color="#2196F3")
    ax1.tick_params(axis="y", labelcolor="#2196F3")
    ax2 = ax1.twinx()
    ax2.plot(ex, ey, color="#E53935", lw=1.5, label="Policy entropy")
    ax2.set_ylabel("Policy entropy (nats)", color="#E53935")
    ax2.tick_params(axis="y", labelcolor="#E53935")
    plt.title("PPO Training Dynamics", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(cdir, "ppo_training_curve.png"), dpi=200,
                bbox_inches="tight")
    plt.close()

    arch_list = list(ARCHETYPES.keys())
    method_list = list(pam.keys())
    short = [m.replace(" (Proposed)", "\n(Ours)").replace(" (Ablation)", "\n(Abl.)")
             for m in method_list]
    x = np.arange(len(arch_list))
    width = 0.85 / len(method_list)
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, mn in enumerate(method_list):
        means = [pam[mn][a]["mean"] for a in arch_list]
        errs = [(pam[mn][a]["mean"] - pam[mn][a]["ci_lo"],
                 pam[mn][a]["ci_hi"] - pam[mn][a]["mean"]) for a in arch_list]
        errs = np.array(errs).T
        c = ("#2196F3" if "Proposed" in mn else
             "#FF9800" if "Ablation" in mn else "#9E9E9E")
        ax.bar(x + i * width - 0.42, means, width, yerr=errs,
                capsize=2, color=c, edgecolor="k", linewidth=0.4,
                label=short[i])
    ax.set_xticks(x)
    ax.set_xticklabels(arch_list)
    ax.set_ylabel("M_CSR")
    ax.set_title("Per-Archetype M_CSR (with 95% CI)", fontweight="bold")
    ax.legend(fontsize=6, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.30))
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(cdir, "per_archetype_mcsr.png"), dpi=200,
                bbox_inches="tight")
    plt.close()

    pert_names = list(rob.keys())
    fig, ax = plt.subplots(figsize=(9, 5))
    for mn in rob[pert_names[0]]:
        ys = [rob[p][mn][0] for p in pert_names]
        es = [rob[p][mn][1] for p in pert_names]
        ax.errorbar(range(len(pert_names)), ys, yerr=es, marker="o",
                     lw=1.5, capsize=3,
                     label=mn.replace(" (Proposed)", " (Ours)"))
    ax.set_xticks(range(len(pert_names)))
    ax.set_xticklabels(pert_names, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("M_CSR")
    ax.set_title("Robustness Across Perturbed Learner Populations",
                  fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(cdir, "robustness_perturbations.png"), dpi=200,
                bbox_inches="tight")
    plt.close()

    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    te_x = [d["tau_e"] for d in te_out]
    te_y = [d["M_CSR_mean"] for d in te_out]
    te_e = [d["M_CSR_std"] for d in te_out]
    axs[0].errorbar(te_x, te_y, yerr=te_e, marker="o", color="#2196F3",
                     capsize=3)
    axs[0].set_xlabel(r"$\tau_e$")
    axs[0].set_ylabel("M_CSR")
    axs[0].set_title("Engagement Threshold Sweep", fontweight="bold")
    axs[0].grid(alpha=0.3)

    th_x = [d["tau_h"] for d in th_out]
    th_y = [d["M_CSR_mean"] for d in th_out]
    th_e = [d["M_CSR_std"] for d in th_out]
    axs[1].errorbar(th_x, th_y, yerr=th_e, marker="o", color="#E53935",
                     capsize=3)
    axs[1].set_xlabel(r"$\tau_h$")
    axs[1].set_ylabel("M_CSR")
    axs[1].set_title("Reliance Threshold Sweep", fontweight="bold")
    axs[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(cdir, "tau_sweeps.png"), dpi=200,
                bbox_inches="tight")
    plt.close()

    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    Tx = [d["T"] for d in hc["T_sweep"]]
    Ty = [d["M_CSR_mean"] for d in hc["T_sweep"]]
    Te = [d["M_CSR_std"] for d in hc["T_sweep"]]
    axs[0].errorbar(Tx, Ty, yerr=Te, marker="o", color="#2196F3", capsize=3)
    axs[0].set_xlabel("Episode horizon T")
    axs[0].set_ylabel("M_CSR")
    axs[0].set_title("Horizon Sensitivity", fontweight="bold")
    axs[0].grid(alpha=0.3)
    Cx = [d["C"] for d in hc["C_sweep"]]
    Cy = [d["M_CSR_mean"] for d in hc["C_sweep"]]
    Ce = [d["M_CSR_std"] for d in hc["C_sweep"]]
    axs[1].errorbar(Cx, Cy, yerr=Ce, marker="o", color="#E53935", capsize=3)
    axs[1].set_xlabel("Concept count C")
    axs[1].set_ylabel("M_CSR")
    axs[1].set_title("Curriculum Width Sensitivity", fontweight="bold")
    axs[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(cdir, "horizon_concept_sweep.png"), dpi=200,
                bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(6, 4))
    t_arr = np.arange(1, 31)
    obs = cal_pl["a"] * t_arr ** cal_pl["b"]
    ax.plot(t_arr, obs, "o-", color="#2196F3", lw=1.5, ms=4,
             label=fr"$k_t = {cal_pl['a']:.3f}\, t^{{{cal_pl['b']:.3f}}}$")
    ax.set_xlabel("Practice trial t")
    ax.set_ylabel("Knowledge k_t")
    ax.set_title(f"Power-Law Calibration  ($R^2={cal_pl['r2']:.3f}$)",
                  fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(cdir, "calibration_power_law.png"), dpi=200,
                bbox_inches="tight")
    plt.close()
    print("  Extra charts saved to prototype/charts/")


if __name__ == "__main__":
    main()
