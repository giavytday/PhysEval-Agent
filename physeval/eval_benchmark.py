"""Standardized benchmark evaluation for PhysEval-Agent.

Measures an LLM's *baseline* scientific coding ability against its *agentic*
ability when the deterministic physics oracle sits in the correction loop:

* **Pass@1** -- single-shot mode (``max_turns=1``): one generation, one sandbox
  execution, one oracle verdict, no repair opportunity.
* **Pass@k** -- agentic mode (default ``k=4``): full Generate -> Execute ->
  Verify -> Correct rollouts where every failure returns a structured JSON
  diagnostic payload to the model.
* **Conservation-drift reduction** -- for trajectories that violate a
  conservation invariant on the first attempt, how far the final attempt
  reduces the governing observable (tracer mass drift, nodal power imbalance,
  steady-state drift, ...) relative to the baseline violation magnitude.

Results are exported as JSONL (one record per task) plus an aggregate summary,
and can be computed against the seed suite or any synthesized
``benchmark_suite.jsonl`` produced by :mod:`physeval.tasks.synthesizer`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from physeval.agent.loop import AsyncPhysEvalLoop, RolloutTask, Trajectory
from physeval.export_dataset import DomainCount, DomainSlice, domain_of
from physeval.sandbox.executor import CodeExecutor
from physeval.tasks.seed_tasks import all_seed_tasks

LOGGER = logging.getLogger("physeval.eval")

#: Metric keys probed in priority order to select the conservation observable.
DRIFT_METRIC_KEYS: Tuple[str, ...] = (
    "tracer_relative_drift",
    "max_nodal_imbalance_mw",
    "steady_state_drift_fraction",
    "max_capture_balance_residual_kg",
    "max_storage_soc_residual_mwh",
)

_DRIFT_EPS = 1e-30


def select_drift_metric(metrics: Dict[str, float]) -> Optional[Tuple[str, float]]:
    """Return ``(name, value)`` of the highest-priority conservation metric."""
    for key in DRIFT_METRIC_KEYS:
        value = metrics.get(key)
        if isinstance(value, (int, float)):
            return key, float(value)
    return None


def drift_reduction(before: float, after: float) -> Optional[float]:
    """Relative reduction of a violation metric (positive = improvement).

    Returns ``None`` when the baseline is degenerate (zero/negative), since a
    ratio against it is meaningless.
    """
    if before <= _DRIFT_EPS:
        return None
    return (before - after) / before


class ModeResult(BaseModel):
    """Outcome of one rollout mode for a single task."""

    success: bool = False
    turns_used: int = 0
    wall_time_s: float = 0.0
    initial_attempt_passed: bool = False
    final_violation_names: List[str] = Field(default_factory=list)
    final_metrics: Dict[str, float] = Field(default_factory=dict)
    error_kind: Optional[str] = None


class TaskEvalRecord(BaseModel):
    """Per-task comparison record (one JSONL line)."""

    task_id: str
    domain: str
    difficulty: str = "n/a"
    baseline: ModeResult
    agentic: ModeResult
    drift_metric_name: Optional[str] = None
    #: Positive values mean the agentic loop reduced violation severity.
    drift_reduction: Optional[float] = None


class BenchmarkSummary(BaseModel):
    """Aggregate statistics across evaluated tasks."""

    n_tasks: int = 0
    pass_at_1_rate: float = 0.0
    pass_at_k_rate: float = 0.0
    k_turns: int = 4
    mean_drift_reduction: Optional[float] = None
    drift_improved_fraction: float = 0.0
    top_initial_violations: List[DomainCount] = Field(default_factory=list)
    domain_slices: Dict[str, DomainSlice] = Field(default_factory=dict)


class EvalRunner:
    """Runs baseline vs. oracle-in-the-loop evaluations for many tasks."""

    def __init__(
        self,
        client: Any,
        *,
        model: str,
        executor: Optional[CodeExecutor] = None,
        max_turns: int = 4,
        temperature: float = 0.0,
    ) -> None:
        if max_turns < 2:
            raise ValueError("max_turns must be >= 2 so agentic mode has room to repair.")
        self.client = client
        self.model = str(model)
        self.executor = executor or CodeExecutor()
        self.max_turns = int(max_turns)
        self.temperature = float(temperature)

    # ------------------------------------------------------------------ #

    def _make_loop(self, max_turns: int) -> AsyncPhysEvalLoop:
        return AsyncPhysEvalLoop(
            self.client,
            model=self.model,
            executor=self.executor,
            max_turns=max_turns,
            temperature=self.temperature,
        )

    @staticmethod
    def _mode_result(trajectory: Trajectory) -> ModeResult:
        steps = trajectory.steps
        executed_steps = [s for s in steps if s.executed]
        initial_passed = bool(executed_steps and executed_steps[0].oracle_passed)
        final_step = trajectory.final_step
        violations: List[str] = []
        metrics: Dict[str, float] = {}
        if final_step is not None and final_step.verification:
            verification = final_step.verification
            violations = [
                str(v.get("name"))
                for v in verification.get("violations", [])
                if v.get("severity") == "FATAL"
            ]
            raw_metrics = verification.get("metrics") or {}
            metrics = {str(k): float(v) for k, v in raw_metrics.items()}
        return ModeResult(
            success=trajectory.success,
            turns_used=trajectory.turns_used,
            wall_time_s=trajectory.wall_time_s,
            initial_attempt_passed=initial_passed,
            final_violation_names=violations,
            final_metrics=metrics,
            error_kind=trajectory.final_error_kind,
        )

    async def eval_task(self, task: RolloutTask) -> TaskEvalRecord:
        """Evaluate *task* under both modes and compute drift reduction."""
        baseline_traj = await self._make_loop(max_turns=1).run(task)
        agentic_traj = await self._make_loop(max_turns=self.max_turns).run(task)

        baseline = self._mode_result(baseline_traj)
        agentic = self._mode_result(agentic_traj)

        before = select_drift_metric(baseline.final_metrics)
        after = select_drift_metric(agentic.final_metrics)
        metric_name: Optional[str] = None
        reduction: Optional[float] = None
        if before is not None and after is not None and before[0] == after[0]:
            metric_name = before[0]
            reduction = drift_reduction(before[1], after[1])

        return TaskEvalRecord(
            task_id=task.id,
            domain=domain_of(task.id),
            difficulty=str(getattr(task, "difficulty", "n/a")),
            baseline=baseline,
            agentic=agentic,
            drift_metric_name=metric_name,
            drift_reduction=reduction,
        )

    # ------------------------------------------------------------------ #

    @staticmethod
    def summarize(records: List[TaskEvalRecord], k_turns: int) -> BenchmarkSummary:
        """Aggregate per-task records into the headline benchmark summary."""
        summary = BenchmarkSummary(n_tasks=len(records), k_turns=k_turns)
        if not records:
            return summary

        p1 = sum(1 for r in records if r.baseline.success)
        pk = sum(1 for r in records if r.agentic.success)
        summary.pass_at_1_rate = p1 / len(records)
        summary.pass_at_k_rate = pk / len(records)

        reductions = [r.drift_reduction for r in records if r.drift_reduction is not None]
        if reductions:
            summary.mean_drift_reduction = sum(reductions) / len(reductions)
        summary.drift_improved_fraction = (
            sum(1 for x in reductions if x is not None and x > 0.0) / len(records)
        )

        violation_counter: Counter[str] = Counter()
        domain_agg: Dict[str, Dict[str, int]] = {}
        for rec in records:
            for name in rec.baseline.final_violation_names or (
                [f"exec:{rec.baseline.error_kind}"] if rec.baseline.error_kind else []
            ):
                violation_counter[name] += 1
            slot = domain_agg.setdefault(rec.domain, {"n": 0, "p1": 0, "pk": 0})
            slot["n"] += 1
            slot["p1"] += int(rec.baseline.success)
            slot["pk"] += int(rec.agentic.success)

        summary.top_initial_violations = [
            DomainCount(name=name, count=count)
            for name, count in violation_counter.most_common(10)
        ]
        summary.domain_slices = {
            dom: DomainSlice(
                tasks=s_["n"],
                pass_at_1=s_["p1"] / s_["n"],
                pass_at_k=s_["pk"] / s_["n"],
            )
            for dom, s_ in sorted(domain_agg.items())
        }
        return summary


def format_eval_report(summary: BenchmarkSummary) -> str:
    """Console rendering of the aggregate benchmark summary."""
    mean_drift = (
        f"{100.0 * summary.mean_drift_reduction:.2f}%"
        if summary.mean_drift_reduction is not None
        else "n/a"
    )
    lines = [
        "=" * 62,
        "PhysEval benchmark summary",
        "=" * 62,
        f"tasks evaluated       : {summary.n_tasks}",
        f"Pass@1 (no feedback)  : {100.0 * summary.pass_at_1_rate:.2f}%",
        f"Pass@{summary.k_turns} (oracle loop) : {100.0 * summary.pass_at_k_rate:.2f}%",
        f"mean drift reduction  : {mean_drift}",
        f"drift improved share  : {100.0 * summary.drift_improved_fraction:.2f}%",
        "",
        "top initial failures:",
    ]
    if summary.top_initial_violations:
        for vc in summary.top_initial_violations:
            lines.append(f"  {vc.count:>6d}  {vc.name}")
    else:
        lines.append("  (none)")
    lines.append("")
    lines.append("per-domain slices:")
    for dom, sl in summary.domain_slices.items():
        lines.append(
            f"  {dom:<10} tasks={sl.tasks:<5d} "
            f"pass@1={100.0 * sl.pass_at_1:6.2f}%  pass@{summary.k_turns}={100.0 * sl.pass_at_k:6.2f}%"
        )
    lines.append("=" * 62)
    return "\n".join(lines)


# ---------------------------------------------------------------------- #
# CLI                                                                    #
# ---------------------------------------------------------------------- #

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="physeval-eval-benchmark",
        description=(
            "Measure baseline Pass@1 vs. agentic Pass@k with physics-oracle "
            "feedback, plus conservation-drift reduction."
        ),
    )
    parser.add_argument("--tasks", default="all", help="Comma-separated seed ids or 'all'.")
    parser.add_argument("--suite", default=None,
                        help="Path to a synthesized benchmark_suite.jsonl "
                             "(overrides --tasks).")
    parser.add_argument(
        "--client",
        default="openai",
        choices=["openai", "mock"],
        help="'openai' uses the OpenAI-compatible API; 'mock' is a deterministic "
             "offline stand-in for smoke tests.",
    )
    parser.add_argument("-o", "--output", default="eval_results.jsonl")
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--max-turns", type=int, default=4)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout-s", type=float, default=20.0)
    parser.add_argument("--mem-limit-mb", type=int, default=None)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    parser.add_argument("--list-tasks", action="store_true")
    return parser.parse_args(argv)


async def run_evaluation(args: argparse.Namespace) -> int:
    """Execute the dual-mode evaluation batch; returns a process exit code."""
    if args.client == "mock":
        from physeval.mock_client import MockChatClient

        client: Any = MockChatClient()
        LOGGER.info("Using deterministic mock client (offline smoke mode).")
    else:
        try:
            from openai import AsyncOpenAI
        except ImportError:
            LOGGER.error(
                "The 'openai' package is required for evaluation. "
                "Install with: pip install 'physeval-agent[llm]'"
            )
            return 2
        api_key = os.environ.get(args.api_key_env) or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            LOGGER.error("No API key found in environment variable %s.", args.api_key_env)
            return 2
        client = AsyncOpenAI(api_key=api_key, base_url=args.base_url)

    tasks: List[RolloutTask]
    if args.suite:
        from physeval.tasks.synthesizer import (
            PhysicsTaskSynthesizer,
            SynthTaskSpec,
        )
        specs = [
            SynthTaskSpec.model_validate(json.loads(line))
            for line in Path(args.suite).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        synth = PhysicsTaskSynthesizer()
        tasks = [synth.spec_to_task(spec) for spec in specs]
    else:
        spec_lower = args.tasks.strip().lower()
        ids = None if spec_lower in {"all", "*"} else [
            s.strip() for s in spec_lower.split(",") if s.strip()
        ]
        try:
            tasks = all_seed_tasks(ids)
        except KeyError as exc:
            LOGGER.error("%s", exc)
            return 2
    if not tasks:
        LOGGER.error("No tasks resolved.")
        return 2

    executor = CodeExecutor(timeout_s=args.timeout_s, mem_limit_mb=args.mem_limit_mb)
    runner = EvalRunner(
        client,
        model=args.model,
        executor=executor,
        max_turns=max(2, args.max_turns),
        temperature=args.temperature,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(max(1, args.concurrency))
    write_lock = asyncio.Lock()
    records: List[TaskEvalRecord] = []

    async def worker(task: RolloutTask) -> None:
        async with semaphore:
            try:
                record = await runner.eval_task(task)
            except Exception as exc:
                LOGGER.exception("[%s] evaluation crashed: %s", task.id, exc)
                return
        records.append(record)
        async with write_lock:
            with output_path.open("a", encoding="utf-8") as fh:
                fh.write(record.model_dump_json() + "\n")
        LOGGER.info(
            "[%s] pass@1=%s pass@%d=%s%s",
            task.id,
            record.baseline.success,
            runner.max_turns,
            record.agentic.success,
            "" if record.drift_reduction is None
            else f" | drift reduced {record.drift_reduction:+.3f}",
        )

    await asyncio.gather(*(worker(t) for t in tasks))
    executor.cleanup()

    summary = runner.summarize(records, k_turns=runner.max_turns)
    print(format_eval_report(summary))
    summary_path = output_path.with_name(output_path.stem + "_summary.json")
    summary_path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")
    print(f"\nrecords -> {output_path}\nsummary -> {summary_path}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point."""
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-7s %(name)s :: %(message)s",
    )
    if args.list_tasks:
        for t in all_seed_tasks():
            print(f"{t.id:32s} [{t.difficulty:6s}] {t.title}")
        return 0
    try:
        return asyncio.run(run_evaluation(args))
    except KeyboardInterrupt:
        print("\nInterrupted; completed per-task records remain valid JSONL.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
