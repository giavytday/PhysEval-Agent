# PhysEval-Agent

**Reinforcement learning with verifiable physical invariants (RLVR) for climate & energy systems.**

Hi, I'm building this in public. PhysEval-Agent is my attempt to answer one question:
**can an LLM agent actually learn to stop hallucinating physics?**

[![status](https://img.shields.io/badge/status-active--development-orange)](#current-scope--whats-next)
[![python](https://img.shields.io/badge/python-3.9%2B-blue)](#quickstart)
[![license](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

## Why this exists

Ask a code LLM to "simulate a power grid" or "run an advection scheme" and you'll usually
get something that *looks* right: clean APIs, confident comments, plausible plots. Then you
check the physics and the bus doesn't balance, the tracer leaks mass every timestep, or the
Courant number is 12. The model optimized for *plausibility*, not conservation laws and
nothing in a standard training loop ever told it otherwise.

RLVR (reinforcement learning with verifiable rewards) changes that: instead of a learned
reward model guessing whether code is good, you use **deterministic oracles as the reward
function**. Conservation laws don't have opinions. Kirchhoff's current law either balances
to within 1e-4 MW or it doesn't.

So the loop here is:

1. An agent writes simulation code for a task with exact numeric parameters.
2. The code runs in a locked-down subprocess sandbox.
3. Deterministic oracles check the exported state against physical invariants.
4. Failures return a *structured JSON diagnostic* not "wrong, try again", but
   "bus `bus3` imbalanced by 12.4 MW at snapshot 17; tolerance is 1e-4".
5. The repair attempt, the diagnostics, and the eventual fix all get logged,
   which means every rollout compiles into PRM step labels, DPO preference pairs,
   and SFT demonstrations. The benchmark *is* the dataset factory.

## System flow

```text
┌────────────────────┐      ┌─────────────────────┐      ┌──────────────────────────┐
│   Task Generator   │      │   Subprocess Sandbox │      │   Physics Oracle         │
│  seeded synthesis  │      │  python -I · rlimits │      │  ⚡ Kirchhoff balance     │
│  grid | climate |  ├─────►│  killpg timeout ·    ├─────►│  🌀 CFL stability         │
│  kinetics (600+)   │ code │  scratch dirs only   │ .nc  │  ⚗️ Arrhenius steady state│
└────────────────────┘      └─────────────────────┘      └───────────┬──────────────┘
                                                                     │ verdict +
                                                                     ▼ violations
┌────────────────────────────────────────────┐      ┌──────────────────────────────┐
│   Dataset Export                           │ ◄────│   Diagnostic Repair Loop     │
│  prm_steps.jsonl · dpo_pairs.jsonl ·       │      │  structured JSON feedback →  │
│  eval_results.jsonl → reports/charts       │      │  targeted patch (≤ k turns)  │
└────────────────────────────────────────────┘      └──────────────────────────────┘
```

## Quickstart

```bash
git clone https://github.com/giavytday/PhysEval-Agent.git
cd PhysEval-Agent

# Core framework, schemas, sandbox, oracles. Heavy deps are lazy-imported.
pip install -e .

# What each stage needs:
pip install -e ".[climate]"   # xarray oracles        (py3.9 OK)
pip install -e ".[grid]"      # PyPSA oracle          (needs py>=3.10 wheels)
pip install -e ".[llm]"       # OpenAI-compatible clients
pip install -e ".[all]"       # everything at once

# Hermetic end-to-end test — no API key, no network, ~2.5 min:
pytest tests/test_end_to_end.py -v        # 6 passed, warnings = errors

# Full pipeline, offline (deterministic mock LLM proves the plumbing):
./run_pipeline.sh --smoke-test            # artifacts under runs/pipeline/

# Live generation against any OpenAI-compatible endpoint:
export OPENAI_API_KEY=sk-...
python -m physeval.run_rollouts --suite tasks/benchmark_suite.jsonl \
    --client openai --model your-model -o trajectories.jsonl --concurrency 16
```

More detail lives in [README archive of CLI examples below](#cli-cheatsheet).

## The three invariant families

Every oracle is a pure function of the exported state file. Same artifact in,
same verdict out — no randomness, no vibes.

### ⚡ Kirchhoff nodal balance (`PyPSAGridOracle`)

At every bus $b$ and snapshot $t$, device injections plus branch flows must vanish:

$$\Bigl|\sum P_{\text{gen}} - \sum P_{\text{load}} + \sum P_{\text{storage}} + \sum_{\text{branches@}b} P^{ij}\Bigr| \le 10^{-4}\ \text{MW}$$

Also checked: line loading $\lvert P_{ij}\rvert \le s_{ij}^{\text{nom}} \cdot s^{\max}_{ij}$,
generator bounds $0 \le P_g \le p_g^{\text{nom}} p^{\max}_g(t)$, and the storage SOC recursion
including converter efficiencies ($\eta_{ch}, \eta_{dis}$), standing losses, inflow and spillage.

### 🌀 CFL stability (`ClimateGridOracle`)

$$C = \frac{u_{\max}\,\Delta t}{\Delta x} \le 1.0, \qquad u_{\max} = \sqrt{u^2+v^2}$$

Plus first-law tracer mass conservation $\left|(M_T - M_0)/M_0\right| \le 10^{-5}$ across
the trajectory, and hard physical bounds: density $\rho > 0$, humidity $q \ge 0$,
temperature $T > 0$ K.

### ⚗️ Arrhenius steady state (`DACKineticsOracle`)

Temperature-swing adsorption cycles with linear driving force kinetics,

$$\dot m = k(T)\,(q_{\text{eq}}(T,p) - m), \qquad k(T) = k_{\text{ref}} e^{-\frac{E_a}{R}\left(\frac{1}{T}-\frac{1}{T_{\text{ref}}}\right)}$$

verified through three lenses: per-cycle capture balance (every kg desorbed must show up in
`captured_cumulative`), cyclic steady state (cycle-to-cycle drift ≤ tolerance), and physical
boundedness ($m \ge 0$, monotone capture, $T > 0$ K).

## Datasets

All three ship full Hugging Face dataset cards (license, taxonomy, schema, safety statement)
via [`physeval.push_to_hub`](physeval/push_to_hub.py):

| Repo | What's inside | Key columns |
|---|---|---|
| `PhysEval-Bench` | 600 seeded tasks (200 grid / 200 climate / 200 kinetics), exact constants embedded | `description`, `requirements`, `artifact_filename`, `oracle_kwargs`, `params` |
| `Phys-PRM-100k` | Step-level process supervision with full oracle error traces | `prompt`, `completion`, `code_span_chars`, `fatal_violations[{name, observed_value, threshold}]`, binary `reward` |
| `Phys-DPO-Pairs` | Failed unverified code paired with its oracle-corrected fix | `prompt` (shared repair context), `rejected`, `chosen`, `rejected_failure_mode`, `turns_to_recover` |

```python
from datasets import load_dataset
bench = load_dataset("<owner>/PhysEval-Bench", split="train")
```

Latest local validation run (600 tasks, deterministic mock LLM — this validates the
*harness*, not any model's physics ability): Pass@1 0.00% → Pass@3 **66.67%**, mean
conservation-drift reduction **100%** on 400 measured tasks, 1,400 PRM steps /
~441k completion tokens, 400 DPO pairs. Grid domain scored 0% here solely because PyPSA
can't execute on this machine's Python 3.9 — see caveats below.

## Current scope — what's next

Honest status, because that's the point of building in public:

**Works today:** the full Generate → Execute → Verify → Correct loop; three oracle families
with unit-tested failure detection; hermetic CI-style integration test (6 stages, zero
warnings, NaN-rejecting JSON parsing); PRM/DPO/SFT export; report charts; HF release tooling;
QLoRA distillation harness (code-complete, untested against real GPUs yet).

**Known limitations:**
- Grid verification needs PyPSA, whose modern wheels require Python ≥ 3.10 — everything else
  runs on 3.9+.
- The sandbox is process-isolated (own session, rlimits, wall-clock kill), **not**
  container-isolated. Fine for research; use a container/gVisor wrapper before pointing it at
  truly adversarial outputs.
- All published metrics so far come from the deterministic mock client. There are no real
  model numbers yet — that's literally the next milestone.

**Next up:** first live rollouts against a real instruct model · PRM reward-model training
on `Phys-PRM-100k` · DPO distillation run · scaling the synthesized suite beyond 600 ·
container-based sandbox option.

## CLI cheatsheet

```bash
python -m physeval.tasks.synthesizer --total 600 --seed 42 -o tasks/benchmark_suite.jsonl
python -m physeval.run_rollouts --suite tasks/benchmark_suite.jsonl \
    --client openai --model <model> -o trajectories.jsonl --concurrency 16
python -m physeval.export_dataset --input trajectories.jsonl --out-dir data/
python -m physeval.eval_benchmark --suite tasks/benchmark_suite.jsonl --output eval_results.jsonl
python -m physeval.generate_report --input eval_results.jsonl --out-dir reports/
python -m physeval.train_distill --stage sft --method qlora --report-to wandb
python -m physeval.train_distill --stage dpo --method qlora --report-to wandb
python -m physeval.push_to_hub --datasets bench,prm,dpo
./run_pipeline.sh --smoke-test | --full [--base-url https://openrouter.ai/api/v1]
```

## License

MIT — see [LICENSE](LICENSE).
