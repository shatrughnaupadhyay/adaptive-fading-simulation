# Adaptive Fading in Agentic Scaffolding -- Simulation Prototype

This repository contains a reproducible simulation of the multi-agent
fading framework described in the paper.  It implements the synthetic
learner model (BEAGLE-inspired), the Student Knowledge Modeling Agent
(SKMA), Engagement Prediction Model (EPM), PPO-trained Tutor Selector
Agent, four baselines, two ablation variants, and all four evaluation
metrics.

## Requirements

- Python 3.10+
- Dependencies listed in `requirements.txt`

### Installation

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Run the full simulation (console output):

```bash
python simulation.py
```

Generate charts alongside console output:

```bash
python simulation.py --charts
```

Use a custom configuration file:

```bash
python simulation.py --config my_config.yaml
```

## Expected Output

The simulation prints six tables to stdout:

| Table | Description |
|-------|-------------|
| TABLE II | Main results: mean +/- std across 5 seeds for M_CSR, H_int, T_eff, K_gain |
| TABLE IIb | Statistical comparisons (Wilcoxon, Cohen's d, bootstrap 95% CI) |
| TABLE III | Ablation study (No-EPM, No-SKMA) |
| TABLE IV | Per-archetype M_CSR breakdown |
| TABLE V | Sensitivity analysis: ZPD estimation noise |
| TABLE VI | Sensitivity analysis: fading threshold sweep |

Key results from the default configuration:

- **Proposed method (AdaptiveFade):** M_CSR = 0.544, H_int = 1.826, T_eff = 0.264, K_gain = 0.106
- Outperforms all four baselines and both ablation variants on M_CSR and T_eff
- Statistically significant improvements confirmed via Wilcoxon signed-rank tests

A saved copy of expected output is in `results.txt` for comparison.

## Configuration

All hyperparameters are centralized in `config.yaml`.  The simulation
loads this file automatically; if it is missing, hardcoded defaults
(matching the paper) are used.

Key parameter groups:

| Section | Parameters | Paper Reference |
|---------|-----------|-----------------|
| `simulation` | N_EPISODES, N_LEARNERS, seeds | Section IV-B |
| `reward` | w1, w2, w3 | Eq. 6 |
| `skma` | alpha, sigma_k | Eq. 2 |
| `epm` | beta1, beta2, sigma_e | Eq. 3 |
| `fading` | theta1, theta2, tau_e, tau_h | Eq. 4--5, Section III-E |
| `learning_dynamics` | phi, scaffold boosts, struggle bounds | Section IV-A |
| `transfer_efficiency` | kappa, rho, sigma_tau | Eq. 7 |
| `archetypes` | mastery/LR/engagement/reliance ranges | Table I |
| `sensitivity` | noise levels, threshold sweep values | Tables V--VI |

## File Structure

```
adaptive-fading-simulation/
  README.md            # this file
  LICENSE              # MIT license
  .gitignore           # excludes .venv, __pycache__, charts
  simulation.py        # main simulation script
  config.yaml          # all hyperparameters (editable)
  requirements.txt     # Python dependencies
  results.txt          # saved reference output
```

Running `python simulation.py --charts` creates a `charts/` directory
with `results_chart.png` and `sensitivity_noise.png`.

## License

MIT -- see [LICENSE](LICENSE) for full text.
