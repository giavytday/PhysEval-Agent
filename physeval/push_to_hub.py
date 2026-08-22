"""Hugging Face dataset & model release pipeline for PhysEval-Agent.

Packages and publishes the three public PhysEval artifacts:

* **PhysEval-Bench** -- the clean 600-task evaluation benchmark (grid,
  climate, and carbon-kinetics scenarios with exact parameter statements and
  machine-checkable requirements).
* **Phys-PRM-100k** -- step-level process-reward supervision harvested from
  sandboxed rollouts, including full oracle error traces (violation names
  with observed/threshold values and metric dictionaries) and binary step
  rewards.
* **Phys-DPO-Pairs** -- preference pairs comparing unverified first-attempt
  code against oracle-corrected self-fixes, keyed on the diagnostic prompt.

For each artifact the script renders a complete dataset card (license,
BibTeX citation, task taxonomy, schema tables, safety statement) and uploads
data + card to the Hub via :mod:`huggingface_hub`. A ``--dry-run`` mode
materializes everything locally without network access; an optional
``--model-dir`` publishes a fine-tuned adapter with a matching model card.

Heavy dependencies are imported lazily::

    pip install 'physeval-agent[hub]'
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

LOGGER = logging.getLogger("physeval.hub")

#: Fixed public repository names (optionally nested under ``--org``).
DATASET_NAMES: Dict[str, str] = {
    "bench": "PhysEval-Bench",
    "prm": "Phys-PRM-100k",
    "dpo": "Phys-DPO-Pairs",
}

_CITATION_BIB = """\
```bibtex
@misc{physeval2026,
  title        = {PhysEval-Agent: Deterministic Scientific Execution and \\
                  Verification for LLM Agents in Climate and Power Systems},
  author       = {PhysEval-Agent Contributors},
  year         = {2026},
  howpublished = {\\url{https://github.com/physeval/physeval-agent}},
}
```
"""


# --------------------------------------------------------------------------- #
# Row builders (dependency-free, unit-testable)                               #
# --------------------------------------------------------------------------- #

def build_benchmark_rows(suite_path: Optional[str | Path]) -> List[Dict[str, Any]]:
    """Load the benchmark suite as plain rows.

    Prefers a synthesized ``benchmark_suite.jsonl`` when available; otherwise
    deterministically regenerates the canonical 600-task suite (seed 42).
    """
    from physeval.tasks.synthesizer import DEFAULT_SUITE_PATH, PhysicsTaskSynthesizer

    resolved = Path(suite_path) if suite_path else DEFAULT_SUITE_PATH
    if resolved.is_file():
        specs = PhysicsTaskSynthesizer.load_suite(resolved)
        provenance = f"synthesized suite loaded from {resolved}"
    else:
        LOGGER.warning(
            "Suite file %s not found; regenerating canonical suite (seed=42).",
            resolved,
        )
        specs = PhysicsTaskSynthesizer(seed=42).synthesize(total=600)
        provenance = "canonical suite regenerated on the fly (seed=42)"

    rows: List[Dict[str, Any]] = []
    for spec in specs:
        record = spec.model_dump()
        record["provenance"] = provenance
        rows.append(record)
    return rows


def build_prm_rows(trajectories_path: str | Path) -> List[Dict[str, Any]]:
    """Flatten rollout trajectories into step-level PRM supervision rows."""
    from physeval.export_dataset import DatasetExporter

    exporter = DatasetExporter(trajectories_path)
    trajectories = exporter.load()
    rows: List[Dict[str, Any]] = []
    for traj in trajectories:
        traj_uid = DatasetExporter.trajectory_uid(traj)
        for rec in exporter.build_prm_records(traj, traj_uid, final_success_turn=None):
            rows.append(rec.model_dump())
    return rows


def build_dpo_rows(trajectories_path: str | Path) -> List[Dict[str, Any]]:
    """Extract preference pairs (failed attempt vs. corrected patch)."""
    from physeval.export_dataset import DatasetExporter

    exporter = DatasetExporter(trajectories_path)
    trajectories = exporter.load()
    rows: List[Dict[str, Any]] = []
    for traj in trajectories:
        traj_uid = DatasetExporter.trajectory_uid(traj)
        for pair in exporter.build_dpo_pairs(traj, traj_uid):
            rows.append(pair.model_dump())
    return rows


# --------------------------------------------------------------------------- #
# Card rendering                                                              #
# --------------------------------------------------------------------------- #

def _taxonomy_table(bench_rows: Sequence[Dict[str, Any]]) -> str:
    """Markdown taxonomy of domain x difficulty counts."""
    by_domain: Counter[str] = Counter()
    by_difficulty: Counter[str] = Counter()
    grid_dom: Counter[str] = Counter()
    for row in bench_rows:
        by_domain[str(row.get("domain"))] += 1
        diff = str(row.get("difficulty"))
        by_difficulty[diff] += 1
        grid_dom[f"{row.get('domain')}/{diff}"] += 1

    lines = [
        "| Cell | Count |",
        "|---|---|",
    ]
    for dom in sorted(by_domain):
        lines.append(f"| domain:`{dom}` | {by_domain[dom]} |")
    for diff in ("easy", "medium", "hard"):
        if by_difficulty[diff]:
            lines.append(f"| difficulty:`{diff}` | {by_difficulty[diff]} |")
    return "\n".join(lines)


def _yaml_header(license_id: str, tags: Sequence[str], size_hint: str) -> str:
    tag_block = "\n".join(f"- {t}" for t in tags)
    return (
        "---\n"
        "language:\n- en\n"
        f"license: {license_id}\n"
        "task_categories:\n- text-generation\n- reinforcement-learning\n"
        "pretty_name: PhysEval\n"
        f"size_categories:\n- {size_hint}\n"
        "tags:\n"
        f"{tag_block}\n"
        "---\n"
    )


def _safety_section() -> str:
    return """\
## Safety & provenance

- All scenario parameters are **synthetic** and generated with seeded RNGs;
  no real infrastructure data is included.
- The corpus contains **no personal data**: prompts describe physics problems
  only. Model outputs are simulation code produced during research rollouts.
- Every supervision label derives from **deterministic physics oracles**
  (conservation laws, bounds, stability criteria) executed inside an isolated
  sandbox -- never from human subjective judgment.
- Rollout code was executed under subprocess isolation with wall-clock and
  memory caps; artifacts were scanned before inclusion.
- Intended use: training and evaluating scientific-computing agents. Users
  remain responsible for validating any generated engineering artifacts.
"""


def _usage_section(repo_id: str, features_hint: str) -> str:
    return f"""\
## Usage

```python
from datasets import load_dataset

ds = load_dataset("{repo_id}", split="train")
print(ds[0])
# {features_hint}
```
"""


def render_benchmark_card(rows: Sequence[Dict[str, Any]], license_id: str,
                          repo_id: str, provenance: str) -> str:
    """Dataset card for **PhysEval-Bench**."""
    header = _yaml_header(
        license_id,
        ["physics", "pypsa", "climate", "adsorption", "code-generation",
         "benchmark", "llm-agents"],
        "10K<n<100K" if len(rows) < 100_000 else "100K<n<1M",
    )
    body = f"""\
{header}
# PhysEval-Bench

A clean evaluation benchmark of **{len(rows)} executable physics-modeling
tasks** for code LLMs and agents, spanning three domains:

| Domain | Description |
|---|---|
| `grid` | PyPSA economic dispatch on randomized multi-bus networks (5-57 buses): nodal power balance, line thermal limits, storage SOC recursions |
| `climate` | Conservative 2D/3D tracer advection(-diffusion) in analytic vortex fields: global mass conservation, CFL stability |
| `kinetics` | Multistage temperature/pressure-swing adsorption cycles: cyclic steady-state mass balance |

Each task embeds *every* numeric constant in its statement (formula-based),
so solutions can be judged deterministically against physics oracles:
nodal imbalances <= 1e-4 MW, tracer drift <= 1e-5 relative, Courant <= 1.0,
per-cycle capture residuals within tolerance.

## Task taxonomy

{_taxonomy_table(rows)}

Provenance: {provenance}.

## Schema

| Column | Type | Description |
|---|---|---|
| `id` | string | Stable task id (`{{domain}}_synth_{{index}}`) |
| `domain` | string | One of `grid`, `climate`, `kinetics` |
| `title` | string | Human-readable task name |
| `description` | string | Full problem statement incl. exact parameters |
| `requirements` | list[string] | Machine-checkable physical requirements |
| `artifact_filename` | string | State file the solution must export |
| `difficulty` | string | `easy` / `medium` / `hard` |
| `tags` | list[string] | Free-form labels |
| `params` | dict[str,float] | Flat sampled-parameter snapshot |
| `oracle_kwargs` | dict | Verifier configuration for reproduction |

## Evaluation protocol

Solutions run inside an isolated sandbox (wall-clock + memory caps); exported
state files are verified by deterministic oracles. Report Pass@1 and
oracle-feedback Pass@k alongside conservation-drift reduction
(`physeval.eval_benchmark`).

## Citation

{_CITATION_BIB}
"""
    return body + _safety_section() + _usage_section(
        repo_id, "a dict with keys id / domain / description / requirements ..."
    )


def render_prm_card(rows: Sequence[Dict[str, Any]], license_id: str,
                    repo_id: str, source: str) -> str:
    """Dataset card for **Phys-PRM-100k**."""
    rewards = Counter(int(r.get("reward", 0)) for r in rows)
    header = _yaml_header(
        license_id,
        ["process-reward-model", "prm", "physics", "code-generation", "rlhf"],
        "10K<n<100K" if len(rows) < 100_000 else "100K<n<1M",
    )
    body = f"""\
{header}
# Phys-PRM-100k

Step-level **process reward supervision** for scientific code generation:
{len(rows)} rollout steps (reward-positive: {rewards.get(1, 0)},
reward-negative: {rewards.get(0, 0)}) collected while agents solved
PhysEval tasks under oracle verification.

Every negative step carries its **full oracle error trace**: violation names,
observed vs. threshold values (e.g., nodal imbalance of 12.3 MW against a
1e-4 MW budget, Courant numbers above 1.0), and scalar metric dictionaries --
the same structured payload used to steer in-loop self-correction.

Source rollouts: `{source}`.

## Schema

| Column | Type | Description |
|---|---|---|
| `uid` | string | Unique step hash |
| `trajectory_uid` | string | Parent rollout identifier |
| `task_id` / `domain` / `model` | string | Provenance fields |
| `turn`, `stage` | int / string | Position in the generate->repair loop |
| `prompt` | string | User message presented at this step |
| `completion` | string | Raw model response |
| `completion_est_tokens` | int | Whitespace-ratio token estimate |
| `code` | string\\|null | Extracted solution code |
| `code_span_chars` | dict\\|null | Char span of `code` within `completion` |
| `exec_ok` | bool | Sandbox execution succeeded |
| `error_kind` | string\\|null | `syntax_error`, `timeout`, ... |
| `oracle_passed` | bool\\|null | Physics invariants held |
| `fatal_violations` | list[dict] | name/severity/observed_value/threshold |
| `metrics` | dict[str,float] | Scalar oracle observables |
| `reward` | float | Binary process label (1.0 valid / 0.0 violated) |
| `is_final_success` | bool | Terminal accepting step of the trajectory |

## Citation

{_CITATION_BIB}
"""
    return body + _safety_section() + _usage_section(
        repo_id, "one dict per rollout step with prompt/completion/reward/oracle_feedback"
    )


def render_dpo_card(rows: Sequence[Dict[str, Any]], license_id: str,
                    repo_id: str, source: str) -> str:
    """Dataset card for **Phys-DPO-Pairs**."""
    header = _yaml_header(
        license_id,
        ["dpo", "preference-data", "physics", "self-correction", "rlhf"],
        "1K<n<10K" if len(rows) < 10_000 else "10K<n<100K",
    )
    body = f"""\
{header}
# Phys-DPO-Pairs

Preference pairs distilled from oracle-in-the-loop self-correction:
**{len(rows)} triplets** where `rejected` is an unverified first-attempt
program that broke a physical invariant (or crashed) and `chosen` is the
model's successful oracle-guided patch, conditioned on the identical
diagnostic repair prompt.

Failure modes span Kirchhoff nodal imbalances, line overloads, generator
capacity breaches, storage SOC recursion errors, tracer mass drift, CFL
violations, and kinetic steady-state failures.

Source rollouts: `{source}`.

## Schema

| Column | Type | Description |
|---|---|---|
| `uid` / `trajectory_uid` | string | Identifiers |
| `task_id` / `domain` / `model` | string | Provenance fields |
| `prompt` | string | Diagnostic repair context (shared condition) |
| `rejected` | string | Failed unverified code |
| `chosen` | string | Oracle-corrected patch |
| `rejected_failure_mode` | string | Violation names or exec error class |
| `rejected_error_kind` | string\\|null | Sandbox error classification |
| `turns_to_recover` | int | Repair distance in turns |
| `rejected_reward` / `chosen_reward` | float | 0.0 / 1.0 |

## Citation

{_CITATION_BIB}
"""
    return body + _safety_section() + _usage_section(
        repo_id, "dict with prompt/chosen/rejected strings for DPOTrainer"
    )


# --------------------------------------------------------------------------- #
# Release manager                                                             #
# --------------------------------------------------------------------------- #

class HubReleaseManager:
    """Prepares and (optionally) pushes PhysEval artifacts to the HF Hub."""

    def __init__(
        self,
        *,
        org: Optional[str] = None,
        private: bool = False,
        dry_run: bool = True,
        token: Optional[str] = None,
        release_dir: str | Path = Path("releases"),
        license_id: str = "mit",
    ) -> None:
        self.org = org
        self.private = private
        self.dry_run = dry_run
        self.token = token
        self.release_dir = Path(release_dir)
        self.license_id = license_id

    def _repo_id(self, name: str) -> str:
        return f"{self.org}/{name}" if self.org else name

    def prepare_local(self, payloads: Dict[str, Tuple[List[Dict[str, Any]], str]]) -> Dict[str, Path]:
        """Write JSONL data + README cards under the local release dir.

        Args:
            payloads: mapping of short key (``bench``/``prm``/``dpo``) to
                ``(rows, card_markdown)``.
        Returns:
            Mapping of short key to the staged folder path.
        """
        staged: Dict[str, Path] = {}
        for key, (rows, card) in payloads.items():
            name = DATASET_NAMES[key]
            folder = self.release_dir / name
            folder.mkdir(parents=True, exist_ok=True)
            data_file = folder / "data.jsonl"
            with data_file.open("w", encoding="utf-8") as fh:
                for row in rows:
                    fh.write(_dump_json_line(row))
                    fh.write("\n")
            (folder / "README.md").write_text(card, encoding="utf-8")
            staged[key] = folder
            LOGGER.info("staged %s: %d rows -> %s", name, len(rows), folder)
        return staged

    def push_all(self, staged: Dict[str, Path]) -> Dict[str, str]:
        """Upload every staged dataset; returns repo ids keyed by short name."""
        if self.dry_run:
            LOGGER.info("dry-run: skipping network upload for %d datasets.", len(staged))
            return {}
        try:
            from huggingface_hub import HfApi
        except ImportError as exc:
            raise ImportError(
                "Missing 'huggingface_hub'; install with: "
                "pip install 'physeval-agent[hub]'"
            ) from exc
        api = HfApi(token=self.token)
        pushed: Dict[str, str] = {}
        for key, folder in staged.items():
            repo_id = self._repo_id(DATASET_NAMES[key])
            api.create_repo(repo_id, repo_type="dataset", private=self.private,
                            exist_ok=True)
            api.upload_folder(folder_path=str(folder), repo_id=repo_id,
                              repo_type="dataset")
            pushed[key] = repo_id
            LOGGER.info("pushed %s (%s)", DATASET_NAMES[key], repo_id)
        return pushed

    def push_model(self, model_dir: str | Path, model_repo: str,
                   base_model: str, method: str) -> str:
        """Publish a fine-tuned adapter directory with a model card."""
        if self.dry_run:
            LOGGER.info("dry-run: skipping model upload for %s", model_dir)
            return ""
        try:
            from huggingface_hub import HfApi
        except ImportError as exc:
            raise ImportError(
                "Missing 'huggingface_hub'; install with: "
                "pip install 'physeval-agent[hub]'"
            ) from exc
        src = Path(model_dir)
        if not src.is_dir():
            raise FileNotFoundError(f"Model directory not found: {src}")
        readme = (
            f"---\nlicense: {self.license_id}\nbase_model: {base_model}\ntags:\n"
            f"- physeval\n- {method}\n---\n\n# PhysEval distilled adapter\n\n"
            f"Trained with {method.upper()} on Phys-PRM-100k / Phys-DPO-Pairs "
            f"(see dataset cards).\n\n{_CITATION_BIB}\n"
        )
        (src / "README.md").write_text(readme, encoding="utf-8")
        api = HfApi(token=self.token)
        repo_id = self._repo_id(model_repo)
        api.create_repo(repo_id, repo_type="model", private=self.private, exist_ok=True)
        api.upload_folder(folder_path=str(src), repo_id=repo_id, repo_type="model")
        LOGGER.info("pushed model %s", repo_id)
        return repo_id


def _dump_json_line(obj: Dict[str, Any]) -> str:
    import json

    return json.dumps(obj, ensure_ascii=False, allow_nan=False, default=str)


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #

def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="physeval-push-to-hub",
        description="Package and publish PhysEval datasets/models to the Hugging Face Hub.",
    )
    parser.add_argument("--trajectories", default="trajectories.jsonl",
                        help="Rollout JSONL feeding the PRM/DPO datasets.")
    parser.add_argument("--suite", default=None,
                        help="benchmark_suite.jsonl (defaults to packaged location).")
    parser.add_argument("--datasets", default="bench,prm,dpo",
                        help="Comma list among bench,prm,dpo.")
    parser.add_argument("--org", default=None, help="Hub organization (optional).")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="Stage cards/data locally without uploading.")
    parser.add_argument("--token-env", default="HF_TOKEN")
    parser.add_argument("--release-dir", default="releases")
    parser.add_argument("--license", dest="license_id", default="mit")
    parser.add_argument("--push-model-dir", default=None)
    parser.add_argument("--model-repo", default="PhysEval-distilled")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point; returns a process exit code."""
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)-7s %(name)s :: %(message)s")

    wanted = [w.strip() for w in args.datasets.split(",") if w.strip()]
    unknown = [w for w in wanted if w not in DATASET_NAMES]
    if unknown:
        print(f"error: unknown datasets {unknown}; choose from {sorted(DATASET_NAMES)}",
              file=sys.stderr)
        return 2

    payloads: Dict[str, Tuple[List[Dict[str, Any]], str]] = {}
    try:
        if "bench" in wanted:
            rows = build_benchmark_rows(args.suite)
            provenance = next((r.get("provenance") for r in rows), "")
            payloads["bench"] = (
                [{k: v for k, v in r.items() if k != "provenance"} for r in rows],
                render_benchmark_card(rows, args.license_id,
                                      f"{args.org or ''}/{DATASET_NAMES['bench']}".strip("/")
                                      if args.org else DATASET_NAMES["bench"],
                                      str(provenance)),
            )
        need_traj = {"prm", "dpo"} & set(wanted)
        if need_traj:
            if not Path(args.trajectories).is_file():
                print(f"error: trajectories file not found: {args.trajectories}",
                      file=sys.stderr)
                return 2
            if "prm" in wanted:
                rows = build_prm_rows(args.trajectories)
                payloads["prm"] = (rows, render_prm_card(
                    rows, args.license_id, DATASET_NAMES["prm"], args.trajectories))
            if "dpo" in wanted:
                rows = build_dpo_rows(args.trajectories)
                payloads["dpo"] = (rows, render_dpo_card(
                    rows, args.license_id, DATASET_NAMES["dpo"], args.trajectories))
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    empty = [k for k, (rows, _) in payloads.items() if not rows]
    if empty:
        print(f"error: datasets {empty} contain zero rows; check source files.",
              file=sys.stderr)
        return 2

    import os

    token = os.environ.get(args.token_env)
    if not args.dry_run and not token:
        print(f"error: no token in ${args.token_env} and --dry-run not set.",
              file=sys.stderr)
        return 2

    manager = HubReleaseManager(
        org=args.org, private=args.private, dry_run=args.dry_run,
        token=token, release_dir=args.release_dir, license_id=args.license_id,
    )
    staged = manager.prepare_local(payloads)
    pushed = manager.push_all(staged)

    if args.push_model_dir:
        manager.push_model(args.push_model_dir, args.model_repo,
                           base_model="see model card", method="qlora")

    mode = "DRY-RUN staged" if args.dry_run else "PUBLISHED"
    for key, folder in staged.items():
        target = pushed.get(key, f"{args.org + '/' if args.org else ''}{DATASET_NAMES[key]}")
        print(f"{mode}: {DATASET_NAMES[key]:<16} rows={len(payloads[key][0]):<7} "
              f"-> {folder if args.dry_run else target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
