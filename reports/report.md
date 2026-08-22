# PhysEval full-suite run (600 tasks, mock-smoke provenance)

*Tasks evaluated:* **600** (0 malformed result lines skipped)

## Headline results

| Scope | Tasks | Pass@1 | Pass@3 (Oracle Self-Correction) |
|---|---:|---:|---:|
| `grid` | 200 | 0.00% | 0.00% |
| `climate` | 200 | 0.00% | 100.00% |
| `kinetics` | 200 | 0.00% | 100.00% |
| **Overall** | **600** | **0.00%** | **66.67%** |

`Pass@3` uses the full Generate → Execute → Verify → Correct loop (max 3 turns) where every failed attempt receives the structured physics-oracle diagnostic payload.

## Conservation-drift reduction

$$\text{Reduction} = \frac{\text{Drift}_{\text{initial}} - \text{Drift}_{\text{final}}}{\text{Drift}_{\text{initial}}} \times 100\%$$

- Measured on **400** task(s) whose baseline violated a conservation invariant.
- Mean relative reduction after oracle-guided correction: **+100.00%**
- Improved: 400 · unchanged/worsened: 0

| Domain | Measured | Mean reduction |
|---|---:|---:|
| `climate` | 200 | +100.00% |
| `kinetics` | 200 | +100.00% |

## Failure modes by domain

| Domain | Failure mode | Occurrences |
|---|---|---:|
| `climate` | `tracer_mass_conservation` | 200 |
| `grid` | `exec:timeout` | 200 |
| `kinetics` | `cyclic_steady_state` | 200 |

*Reading guide:* `nodal_power_balance` denotes Kirchhoff nodal imbalance beyond 1e-4 MW; `cfl_stability` denotes Courant numbers above 1.0; `exec:*` rows are sandbox failures (syntax, imports, timeouts) rather than physics violations.

## Methodology

- Identical base model, temperature 0, and sandbox budgets across modes.
- Baseline = single-shot generation (max 1 turn); agentic = oracle-in-
  the-loop repair up to the stated turn budget.
- Conservation observables are selected deterministically per task:
  tracer mass drift (climate), nodal power imbalance (grid), or
  cyclic steady-state drift / capture residual (kinetics).
