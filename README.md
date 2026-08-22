# PhysEval-Agent: Reinforcement Learning with Verifiable Physical Invariants for Climate & Energy Systems

[![CI](https://img.shields.io/badge/CI-e2e--smoke-green)](#verification)
[![python](https://img.shields.io/badge/python-3.9%2B-blue)](#quickstart)
[![license](https://img.shields.io/badge/license-MIT-lightgrey)](#license)

**PhysEval-Agent** is an open-source, deterministic scientific execution and
verification framework for training and benchmarking LLM agents on climate and
power-systems modeling. Generated simulation code is executed inside an
isolated subprocess sandbox, its exported state artifacts are checked against
*exact* physical invariants (Kirchhoff's law, Courant–Friedrichs–Lewy
stability, Arrhenius cyclic steady state), and structured violation payloads
are fed back to the model for multi-turn self-correction. Every rollout is
logged as a dense trajectory that compiles directly into PRM / DPO / SFT
training data.

---

## Architecture

```text
                        ┌─────────────────────────────────────────────┐
                        │  Task Synthesis (PhysicsTaskSynthesizer)    │
                        │  600 parameterized scenarios                │
                        │  grid | climate | kinetics                  │
                        └──────────────────────┬──────────────────────┘
                                               │ task spec (exact constants)
                                               ▼
┌──────────────────┐   code   ┌──────────────────────────┐   state.nc
│  LLM Agent       │ ───────► │  Subprocess Sandbox      │ ─────────┐
│  (any OpenAI-    │ ◄─────── │  python -I · rlimits     │          │
│   compatible)    │  repair  │  killpg timeout · mock   │          ▼
└──────────────────┘  context └──────────────────────────┘   ┌─────────────────────────────┐
        ▲                                                    │  Oracle Invariant           │
        │  structured JSON diagnostics                       │  Verification               │
        │  (violations + exact metrics)                      │                             │
        │                                                    │  ⚡ Kirchhoff nodal balance  │
        │                                                    │  🌀 CFL stability            │
        │                                                    │  ⚗️  Arrhenius steady state  │
        │                                                    └──────────────┬──────────────┘
        │                                                                   │ pass / fail
        │            ┌────────────────────────────────────────┐             │
        └────────────│  Multi-Turn Self-Correction Loop       │ ◄───────────┘
                     │  Generate → Execute → Verify → Correct │
                     │  (max_turns, default 4)                │
                     └───────────────────┬────────────────────┘
                                         │ Trajectory (per-turn evidence)
                                         ▼
        ┌──────────────────────────────────────────────────────────────┐
        │  Data flywheel                                               │
        │  prm_steps.jsonl · dpo_pairs.jsonl · eval_results.jsonl      │
        │      → distillation (SFT/DPO + LoRA/QLoRA)                   │
        │      → HF Hub release (PhysEval-Bench, Phys-PRM-100k, ...)   │
        │      → markdown + PNG performance reports                    │
        └──────────────────────────────────────────────────────────────┘
```

---

## Quickstart

### Installation

```bash
git clone https://github.com/physeval/physeval-agent.git
cd physeval-agent

# Core framework (schemas, sandbox, oracles)
pip install -e .

# Domain extras
pip install -e ".[grid,climate]"     # PyPSA + xarray verifiers
pip install -e ".[llm]"              # OpenAI-compatible rollouts
pip install -e ".[all]"              # everything scientific + LLM

# Optional stages
pip install -e ".[train]"            # TRL / PEFT distillation
pip install -e ".[hub]"              # Hugging Face releases
pip install -e ".[report]"           # matplotlib charts
```

Python **3.9+**; heavy dependencies are lazy-imported per stage.

### One-command pipeline (hermetic smoke test, no API key required)

```bash
./run_pipeline.sh --smoke-test
# 5 steps: synthesize(3) → rollouts(mock) → export → evaluate → report
# Artifacts land under runs/pipeline/
```

Full-scale run against a real model:

```bash
export OPENAI_API_KEY=sk-...
./run_pipeline.sh --full --model gpt-4o-mini --concurrency 4
```

### Step-by-step

```bash
# 1. Synthesize a benchmark suite (deterministic, seeded)
python -m physeval.tasks.synthesizer --total 600 --seed 42 \
       --out physeval/tasks/benchmark_suite.jsonl

# 2. Batch rollouts with oracle-in-the-loop self-correction
python -m physeval.run_rollouts --suite physeval/tasks/benchmark_suite.jsonl \
       --model gpt-4o-mini --concurrency 4 --max-turns 4 \
       --output trajectories.jsonl

# 3. Export PRM step labels + DPO preference pairs
python -m physeval.export_dataset --trajectories trajectories.jsonl \
       --out-dir data/

# 4. Standardized evaluation: Pass@1 vs Pass@k + drift reduction
python -m physeval.eval_benchmark --suite physeval/tasks/benchmark_suite.jsonl \
       --output eval_results.jsonl

# 5. Markdown tables + high-resolution charts
python -m physeval.generate_report --results eval_results.jsonl --out-dir reports/

# 6. Local distillation (QLoRA on Qwen2.5-Coder)
python -m physeval.train_distill --stage sft  --method qlora \
       --base-model Qwen/Qwen2.5-Coder-7B-Instruct --report-to wandb
python -m physeval.train_distill --stage dpo  --method qlora \
       --data data/dpo_pairs.jsonl --report-to wandb

# 7. Publish datasets to the Hugging Face Hub (dry-run by default)
python -m physeval.push_to_hub --datasets bench,prm,dpo --org my-org
```

---

## Physics Invariant Families

Every oracle is a pure function of the exported state file: no randomness,
no network I/O, identical artifacts ⇒ identical verdicts.

### Family A — Power Grids (`PyPSAGridOracle`)

**Nodal active-power balance (Kirchhoff's current law).** For every bus $b$
and snapshot $t$:

$$\Bigl|\underbrace{\textstyle\sum_{g\in b} P_g(t) - \sum_{\ell\in b} P_\ell(t) + \sum_{s\in b} P_s(t)}_{\text{device injections}} \;+\; \underbrace{\sum_{b\in\cdot} P^{ij}(t)}_{\text{branch flows}}\Bigr| \;\le\; 10^{-4}\ \text{MW}$$

Additional asserted constraints:

| Invariant | Condition |
|---|---|
| Line thermal limit | $\lvert P_{ij}(t)\rvert \le s^{\max}_{ij} = s^{\text{nom}}_{ij}\,\bar{s}_{ij}$ |
| Generator capacity | $0 - \epsilon \le P_g(t) \le p^{\text{nom}}_g \, p^{\max}_g(t) + \epsilon$ |
| Storage SOC recursion | $e_t = e_{t-1}(1-\sigma\Delta t) \;+\; \eta_{\text{ch}} \max(-P_t,0)\Delta t \;-\; \frac{\max(P_t,0)}{\eta_{\text{dis}}}\Delta t \;+\; (\text{inflow}-\text{spill})\Delta t$ |

### Family B — Climate Dynamics (`ClimateGridOracle`)

**Courant–Friedrichs–Lewy numerical stability.**

$$C \;=\; \frac{u_{\max}\,\Delta t}{\Delta x} \;\le\; 1.0$$

with $u_{\max}=\sqrt{u^2+v^2}$ evaluated over the full wind-magnitude field.

**First-law global tracer mass conservation** between the first and last frame:

$$\left|\frac{M_T - M_0}{M_0}\right| \le 10^{-5}, \qquad M_t = \sum_{x,y,z} c(x,y,z,t)\,w(x,y,z)$$

**Physical bounds:** density $\rho > 0$, specific humidity $q \ge 0$,
absolute temperature $T > 0\,$K (implausible-range warnings outside 50–500 K).

### Family C — Carbon Kinetics (`DACKineticsOracle`)

**Arrhenius-corrected linear driving force** on the sorbent bed loading $m$:

$$\frac{dm}{dt} = k(T)\,\bigl(q_{\text{eq}}(T,p) - m\bigr), \qquad k(T) = k_{\text{ref}} \exp\!\Bigl[-\frac{E_a}{R}\Bigl(\frac{1}{T} - \frac{1}{T_{\text{ref}}}\Bigr)\Bigr]$$

with Langmuir equilibrium affinity following van't Hoff: $b(T) = b_0 \exp\bigl[\tfrac{\Delta H}{R}\bigl(\tfrac{1}{T} - \tfrac{1}{T_{\text{ref}}}\bigr)\bigr]$ (exothermic ⇒ affinity decreases with $T$).

**Cyclic steady-state tolerance.** With cycle throughput scale
$\Phi_c = \max(\text{captured}_c,\ \text{desorbed}_c)$:

$$\frac{|m^{\text{end}}_c - m^{\text{start}}_c|}{\Phi_c} \le \tau_{\text{ss}}, \qquad \frac{|\,\text{captured}_c - \text{captured}_{c-1}\,|}{\text{captured}_{c-1}} \le \tau_{\text{ss}}$$

**Per-cycle capture balance:** every kilogram desorbed during regeneration
must appear in `captured_cumulative`:

$$\Bigl|\,\Delta\,\text{captured}\big|_{c} - \sum_{t \in c}\max(-\Delta m_t, 0)\Bigr| \le \epsilon_{\text{bal}}$$

plus physical boundedness $m_t \ge 0$, monotone capture, $T > 0$ K.

---

## Benchmark Tasks

| Task id | Domain | Difficulty | Verifier | Core challenge |
|---|---|---|---|---|
| `grid_24h_curtailment` | PyPSA | hard | `PyPSAGridOracle` | Economic dispatch of a 5-bus wind+battery ring without line overloads |
| `tracer_advection_2d` | xarray | medium | `ClimateGridOracle` | Flux-form advection of a Gaussian blob in a vortex field, mass-exact |
| `direct_air_capture_kinetics` | kinetics | hard | `DACKineticsOracle` | Multistage TSA cycles to cyclic steady state |
| `*_synth_*` (600) | all | easy→hard | all three | Seeded parameterized suite (`tasks/benchmark_suite.jsonl`) |

---

## Dataset & Benchmark Cards (Hugging Face Hub)

Released via [`physeval.push_to_hub`](physeval/push_to_hub.py); each artifact
ships a full dataset card (license, citation, taxonomy, schema, safety).

| Repository | Contents | Row schema highlights |
|---|---|---|
| **`PhysEval-Bench`** | The clean 600-task evaluation benchmark | `id`, `domain`, `description` (all constants embedded), `requirements`, `artifact_filename`, `difficulty`, `params`, `oracle_kwargs` |
| **`Phys-PRM-100k`** | Step-level process reward supervision with oracle error traces | `prompt`, `completion`, `code_span_chars`, `fatal_violations[{name, observed_value, threshold}]`, `metrics`, binary `reward ∈ {0.0, 1.0}` |
| **Phys-DPO-Pairs** | Preference pairs: unverified first attempt vs. oracle-corrected patch | `prompt` (shared diagnostic condition), `rejected`, `chosen`, `rejected_failure_mode`, `turns_to_recover` |

Safety & provenance statement (included in every card): fully synthetic seeded
parameters, zero personal data, labels derived exclusively from deterministic
physics oracles executed in isolated sandboxes.

```python
from datasets import load_dataset
bench = load_dataset("your-org/PhysEval-Bench", split="train")
prm   = load_dataset("your-org/Phys-PRM-100k", split="train")
dpo   = load_dataset("your-org/Phys-DPO-Pairs", split="train")
```

---

## Repository Layout

```text
physeval/
├── oracle/            # deterministic verifiers
│   ├── base.py            # Pydantic v2 schemas + BasePhysicsOracle ABC
│   ├── grid_invariants.py # Kirchhoff · thermal · capacity · SOC recursion
│   ├── climate_invariants.py  # tracer mass · bounds · CFL
│   └── (seed_tasks.py hosts DACKineticsOracle for TSA cycles)
├── sandbox/executor.py    # subprocess isolation, timeouts, artifact discovery
├── agent/
│   ├── prompts.py         # generation contract + strict-JSON repair prompts
│   ├── loop.py            # AsyncPhysEvalLoop (Generate→Execute→Verify→Correct)
│   └── ...
├── tasks/synthesizer.py   # 600-task seeded generator
├── mock_client.py         # offline deterministic LLM stand-in
├── run_rollouts.py        # async batch rollout CLI → JSONL
├── export_dataset.py      # PRM + DPO formatters + stats
├── eval_benchmark.py      # Pass@1 vs Pass@k + drift reduction
├── generate_report.py     # markdown tables + PNG charts
├── train_distill.py       # SFT/DPO harness (LoRA/QLoRA, W&B/TensorBoard)
└── push_to_hub.py         # dataset/model release pipeline
tests/test_end_to_end.py   # hermetic integration suite
run_pipeline.sh            # 5-step master runner (--smoke-test / --full)
```

## Verification

The hermetic end-to-end suite drives one task per domain through every stage
using real CLI entry points and asserts schema conformance, NaN-free JSONL,
reward sanity, pass-rate invariants, and chart generation — with warnings
escalated to errors (`filterwarnings = ["error"]`):

```bash
pytest tests/test_end_to_end.py -v
# 6 passed — rollouts · export · evaluation · report · shell runner

# full unit + functional matrix
pytest tests/ -q        # plus external unit suites if present
```

## Citation

If you use PhysEval-Agent, please cite:

```bibtex
@misc{physeval2026,
  title        = {PhysEval-Agent: Reinforcement Learning with Verifiable
                  Physical Invariants for Climate \& Energy Systems},
  author       = {PhysEval-Agent Contributors},
  year         = {2026},
  howpublished = {\url{https://github.com/physeval/physeval-agent}},
}
```

## License

MIT — see [pyproject.toml](pyproject.toml).
