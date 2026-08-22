"""Dataset formatting pipelines for PhysEval rollout trajectories.

Parses ``trajectories.jsonl`` (the output of :mod:`physeval.run_rollouts`)
and produces two Hugging Face-ready exports plus a statistics report:

* ``prm_steps.jsonl`` -- step-level *process supervision* records. Each row
  carries the full prompt/completion pair, the character span of the embedded
  solution code, structured oracle feedback (violation names with exact
  observed/threshold values and metric dictionaries), and a binary step
  reward: ``1.0`` when the sandbox run succeeded **and** every physical
  invariant held, ``0.0`` otherwise.
* ``dpo_pairs.jsonl`` -- preference pairs pairing a *failed* initial
  generation (rejected) with the *successful* self-corrected patch (chosen),
  conditioned on the exact diagnostic prompt that triggered the repair.
* ``stats.json`` / console report -- pass@1 vs pass@k rates, most common
  fatal violation types, average recovery turns, and per-domain slices.

Malformed lines are skipped and counted rather than aborting the export.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from pydantic import BaseModel, Field

LOGGER = logging.getLogger("physeval.export")

#: Estimated tokens per character for budget reporting (GPT-style tokenizer).
_CHARS_PER_TOKEN = 4.0

#: Known task-id prefixes used to slice statistics by physics domain.
_DOMAIN_PREFIXES: Dict[str, str] = {
    "grid": "grid",
    "climate": "climate",
    "kinetics": "kinetics",
    "tracer_advection_2d": "climate",
    "grid_24h_curtailment": "grid",
    "direct_air_capture_kinetics": "kinetics",
}


def domain_of(task_id: str) -> str:
    """Best-effort mapping from a task id to its physics domain."""
    lowered = task_id.lower()
    for prefix, domain in _DOMAIN_PREFIXES.items():
        if lowered.startswith(prefix):
            return domain
    return "other"


class FatalViolation(BaseModel):
    """Compact violation evidence retained in PRM rows."""

    name: str
    severity: str
    observed_value: float
    threshold: float


class PRMStepRecord(BaseModel):
    """One step-level process-supervision example."""

    uid: str
    trajectory_uid: str
    task_id: str
    domain: str
    model: str
    turn: int
    stage: str
    prompt: str
    completion: str
    completion_est_tokens: int
    code: Optional[str] = None
    code_span_chars: Optional[Dict[str, int]] = None
    exec_ok: bool = False
    error_kind: Optional[str] = None
    oracle_passed: Optional[bool] = None
    fatal_violations: List[FatalViolation] = Field(default_factory=list)
    metrics: Dict[str, float] = Field(default_factory=dict)
    reward: float = 0.0
    is_final_success: bool = False

    @property
    def attempted_execution(self) -> bool:
        """True when this step produced code that entered the sandbox."""
        return self.code is not None


class DPOPairRecord(BaseModel):
    """One preference pair: failed attempt vs. successful self-correction."""

    uid: str
    trajectory_uid: str
    task_id: str
    domain: str
    model: str
    prompt: str  # diagnostic repair context shown before the chosen patch
    rejected: str
    chosen: str
    rejected_failure_mode: str
    rejected_error_kind: Optional[str] = None
    turns_to_recover: int
    rejected_reward: float = 0.0
    chosen_reward: float = 1.0


class DomainCount(BaseModel):
    name: str
    count: int


class DomainSlice(BaseModel):
    tasks: int = 0
    pass_at_1: float = 0.0
    pass_at_k: float = 0.0


class DatasetStats(BaseModel):
    """Aggregate report over an exported dataset."""

    total_lines: int = 0
    malformed_lines: int = 0
    total_trajectories: int = 0
    steps_total: int = 0
    prm_records: int = 0
    dpo_pairs: int = 0
    pass_at_1_rate: float = 0.0
    pass_at_k_rate: float = 0.0
    k_max_turns: int = 4
    initially_failed: int = 0
    recovered: int = 0
    recovery_rate: float = 0.0
    avg_recovery_turns: float = 0.0
    mean_wall_time_s: float = 0.0
    exec_error_distribution: Dict[str, int] = Field(default_factory=dict)
    top_fatal_violations: List[DomainCount] = Field(default_factory=list)
    domain_slices: Dict[str, DomainSlice] = Field(default_factory=dict)


class ExportPaths(BaseModel):
    prm_steps: Path
    dpo_pairs: Path
    stats_json: Path


class DatasetExporter:
    """Transforms rollout trajectories into training-ready datasets."""

    def __init__(
        self,
        trajectories_path: str | Path,
        out_dir: Optional[str | Path] = None,
        *,
        default_k: int = 4,
    ) -> None:
        """Configure the exporter.

        Args:
            trajectories_path: JSONL produced by ``run_rollouts.py``.
            out_dir: Destination directory; defaults to beside the input.
            default_k: Pass@k budget reported when no trajectory sets one.
        """
        self.trajectories_path = Path(trajectories_path)
        self.out_dir = Path(out_dir) if out_dir else self.trajectories_path.parent / "dataset_export"
        self.default_k = max(1, int(default_k))
        self._trajs: List[Dict[str, Any]] = []
        self.malformed_lines = 0

    # ------------------------------------------------------------------ #
    # Loading                                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def iter_raw_lines(path: Path) -> Iterator[Tuple[int, str]]:
        with path.open("r", encoding="utf-8") as fh:
            yield from enumerate(fh, start=1)

    def load(self) -> List[Dict[str, Any]]:
        """Parse and validate trajectory records; skips malformed lines."""
        if not self.trajectories_path.is_file():
            raise FileNotFoundError(f"Trajectory file not found: {self.trajectories_path}")
        trajs: List[Dict[str, Any]] = []
        for lineno, line in self.iter_raw_lines(self.trajectories_path):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, dict) or "task_id" not in record or "steps" not in record:
                    raise ValueError("missing required keys 'task_id'/'steps'")
                trajs.append(record)
            except (json.JSONDecodeError, ValueError) as exc:
                self.malformed_lines += 1
                LOGGER.warning("Skipping malformed line %d: %s", lineno, exc)
        self._trajs = trajs
        return trajs

    # ------------------------------------------------------------------ #
    # Record builders                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def trajectory_uid(record: Dict[str, Any]) -> str:
        basis = f"{record.get('task_id', '?')}:{record.get('model', '?')}"
        first_prompt = ""
        steps = record.get("steps") or []
        if steps and isinstance(steps[0], dict):
            first_prompt = str(steps[0].get("prompt") or "")
        digest = hashlib.sha1(f"{basis}:{first_prompt}".encode()).hexdigest()[:10]
        return f"{record.get('task_id', 'task')}:{digest}"

    @staticmethod
    def _code_span(completion: str, code: Optional[str]) -> Optional[Dict[str, int]]:
        if not code or code not in completion:
            return None
        start = completion.find(code)
        return {"start": start, "end": start + len(code)}

    @staticmethod
    def _fatal(violations: Any) -> List[FatalViolation]:
        out: List[FatalViolation] = []
        for v in violations or []:
            if isinstance(v, dict):
                out.append(
                    FatalViolation(
                        name=str(v.get("name", "unknown")),
                        severity=str(v.get("severity", "FATAL")),
                        observed_value=float(v.get("observed_value") or 0.0),
                        threshold=float(v.get("threshold") or 0.0),
                    )
                )
        return out

    @classmethod
    def build_prm_records(
        cls, record: Dict[str, Any], traj_uid: str, final_success_turn: Optional[int]
    ) -> List[PRMStepRecord]:
        """Convert one trajectory into step-level supervision rows."""
        rows: List[PRMStepRecord] = []
        base = {
            "trajectory_uid": traj_uid,
            "task_id": str(record.get("task_id", "?")),
            "domain": domain_of(str(record.get("task_id", ""))),
            "model": str(record.get("model", "?")),
        }
        for idx, step in enumerate(record.get("steps") or []):
            if not isinstance(step, dict):
                continue
            completion = str(step.get("raw_response") or "")
            code = step.get("extracted_code")
            code = code.strip() if isinstance(code, str) and code.strip() else None
            execution = step.get("execution") or {}
            verification = step.get("verification")
            exec_ok = bool(execution.get("ok")) if isinstance(execution, dict) else False
            error_kind = (
                str(execution.get("error_kind"))
                if isinstance(execution, dict) and execution.get("error_kind")
                else ("unparseable" if code is None and not completion.startswith("<llm_error>") else "llm_error")
            )
            oracle_passed: Optional[bool] = None
            fatal: List[FatalViolation] = []
            metrics: Dict[str, float] = {}
            if isinstance(verification, dict):
                oracle_passed = bool(verification.get("passed"))
                fatal = cls._fatal(verification.get("violations"))
                raw_metrics = verification.get("metrics")
                if isinstance(raw_metrics, dict):
                    metrics = {str(k): float(v) for k, v in raw_metrics.items()}
            turn = int(step.get("turn") or idx + 1)
            passed_step = bool(exec_ok and oracle_passed)
            rows.append(
                PRMStepRecord(
                    uid=hashlib.sha1(f"{traj_uid}|{turn}|{idx}".encode()).hexdigest()[:16],
                    turn=turn,
                    stage=str(step.get("stage") or "generate"),
                    prompt=str(step.get("prompt") or ""),
                    completion=completion,
                    completion_est_tokens=max(1, round(len(completion) / _CHARS_PER_TOKEN)),
                    code=code,
                    code_span_chars=cls._code_span(completion, code) if code else None,
                    exec_ok=exec_ok,
                    error_kind=None if passed_step else error_kind,
                    oracle_passed=oracle_passed,
                    fatal_violations=fatal,
                    metrics=metrics,
                    reward=1.0 if passed_step else 0.0,
                    is_final_success=(
                        passed_step and final_success_turn is not None and turn == final_success_turn
                    ),
                    **base,
                )
            )
        return rows

    @staticmethod
    def build_dpo_pairs(
        record: Dict[str, Any], traj_uid: str
    ) -> List[DPOPairRecord]:
        """Pair the first failing coded attempt with the first later success."""
        steps = [s for s in (record.get("steps") or []) if isinstance(s, dict)]
        coded_steps = [
            s for s in steps
            if isinstance(s.get("extracted_code"), str) and s["extracted_code"].strip()
        ]
        if len(coded_steps) < 2:
            return []

        def _passed(step: Dict[str, Any]) -> bool:
            execution = step.get("execution") or {}
            verification = step.get("verification")
            return bool(execution.get("ok")) and bool(
                isinstance(verification, dict) and verification.get("passed")
            )

        fail_idx: Optional[int] = next(
            (i for i, s in enumerate(coded_steps) if not _passed(s)), None
        )
        if fail_idx is None:
            return []
        succ_idx_list = [
            i for i, s in enumerate(coded_steps) if i > fail_idx and _passed(s)
        ]
        if not succ_idx_list:
            return []

        bad, good = coded_steps[fail_idx], coded_steps[succ_idx_list[0]]
        good_code = str(good["extracted_code"]).strip()
        bad_code = str(bad["extracted_code"]).strip()
        if good_code == bad_code:
            return []

        execution = bad.get("execution") or {}
        verification = bad.get("verification")
        error_kind = execution.get("error_kind") if isinstance(execution, dict) else None
        names: List[str] = []
        if isinstance(verification, dict):
            names = [
                str(v.get("name"))
                for v in verification.get("violations") or []
                if v.get("severity") == "FATAL"
            ]
        failure_mode = ",".join(names) if names else f"exec:{error_kind or 'unknown'}"
        base_meta = {
            "trajectory_uid": traj_uid,
            "task_id": str(record.get("task_id", "?")),
            "domain": domain_of(str(record.get("task_id", ""))),
            "model": str(record.get("model", "?")),
        }
        return [
            DPOPairRecord(
                uid=hashlib.sha1(f"{traj_uid}|dpo".encode()).hexdigest()[:16],
                prompt=str(good.get("prompt") or ""),
                rejected=bad_code,
                chosen=good_code,
                rejected_failure_mode=failure_mode,
                rejected_error_kind=(str(error_kind) if error_kind else None),
                turns_to_recover=int(good.get("turn") or 0) - int(bad.get("turn") or 0),
                **base_meta,
            )
        ]

    # ------------------------------------------------------------------ #
    # Statistics                                                         #
    # ------------------------------------------------------------------ #

    def compute_stats(self, trajs: List[Dict[str, Any]]) -> DatasetStats:
        """Aggregate pass rates, recovery behavior, and violation frequency."""
        stats = DatasetStats(total_lines=self.malformed_lines + len(trajs),
                             malformed_lines=self.malformed_lines,
                             total_trajectories=len(trajs))
        first_pass = 0
        any_pass = 0
        wall_times: List[float] = []
        recovery_turns: List[int] = []
        violation_counter: Counter[str] = Counter()
        error_counter: Counter[str] = Counter()
        domain_agg: Dict[str, Dict[str, int]] = {}

        for record in trajs:
            steps = [s for s in (record.get("steps") or []) if isinstance(s, dict)]
            stats.steps_total += len(steps)
            traj_uid = self.trajectory_uid(record)
            prm_rows = self.build_prm_records(record, traj_uid, final_success_turn=None)
            stats.prm_records += len(prm_rows)

            executed = [r for r in prm_rows if r.attempted_execution]
            k_budget = int(record.get("max_turns") or self.default_k)
            stats.k_max_turns = max(stats.k_max_turns, k_budget)

            dom = domain_of(str(record.get("task_id", "")))
            slot = domain_agg.setdefault(dom, {"n": 0, "p1": 0, "pk": 0})
            slot["n"] += 1

            success = bool(record.get("success"))
            any_pass += int(success)
            slot["pk"] += int(success)
            wall_times.append(float(record.get("wall_time_s") or 0.0))

            first_exec_pass = bool(executed and executed[0].reward == 1.0)
            first_pass += int(first_exec_pass)
            slot["p1"] += int(first_exec_pass)

            if not first_exec_pass:
                stats.initially_failed += 1
                if success:
                    stats.recovered += 1

            for row in prm_rows:
                if row.fatal_violations:
                    for v in row.fatal_violations:
                        violation_counter[v.name] += 1
                if row.error_kind and row.error_kind not in ("none",):
                    error_counter[row.error_kind] += 1

            for pair in self.build_dpo_pairs(record, traj_uid):
                stats.dpo_pairs += 1
                recovery_turns.append(pair.turns_to_recover)

        total = max(stats.total_trajectories, 1)
        stats.pass_at_1_rate = first_pass / total
        stats.pass_at_k_rate = any_pass / total
        stats.recovery_rate = (
            stats.recovered / stats.initially_failed if stats.initially_failed else 0.0
        )
        stats.avg_recovery_turns = (
            sum(recovery_turns) / len(recovery_turns) if recovery_turns else 0.0
        )
        stats.mean_wall_time_s = sum(wall_times) / len(wall_times) if wall_times else 0.0
        stats.exec_error_distribution = dict(error_counter.most_common())
        stats.top_fatal_violations = [
            DomainCount(name=name, count=count)
            for name, count in violation_counter.most_common(10)
        ]
        stats.domain_slices = {
            dom: DomainSlice(
                tasks=slice_["n"],
                pass_at_1=slice_["p1"] / slice_["n"] if slice_["n"] else 0.0,
                pass_at_k=slice_["pk"] / slice_["n"] if slice_["n"] else 0.0,
            )
            for dom, slice_ in sorted(domain_agg.items())
        }
        return stats

    # ------------------------------------------------------------------ #
    # Export                                                             #
    # ------------------------------------------------------------------ #

    def export(self, *, write: bool = True) -> Tuple[DatasetStats, Optional[ExportPaths]]:
        """Run the full pipeline: parse -> build records -> write files."""
        trajs = self.load()
        stats = self.compute_stats(trajs)
        paths: Optional[ExportPaths] = None

        if write:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            prm_path = self.out_dir / "prm_steps.jsonl"
            dpo_path = self.out_dir / "dpo_pairs.jsonl"
            stats_path = self.out_dir / "stats.json"

            n_prm = n_dpo = 0
            with prm_path.open("w", encoding="utf-8") as prm_fh, \
                    dpo_path.open("w", encoding="utf-8") as dpo_fh:
                for record in trajs:
                    traj_uid = self.trajectory_uid(record)
                    for row in self.build_prm_records(record, traj_uid, final_success_turn=None):
                        prm_fh.write(row.model_dump_json() + "\n")
                        n_prm += 1
                    for pair in self.build_dpo_pairs(record, traj_uid):
                        dpo_fh.write(pair.model_dump_json() + "\n")
                        n_dpo += 1
            stats.prm_records = n_prm
            stats.dpo_pairs = n_dpo
            stats_path.write_text(json.dumps(stats.model_dump(), indent=2), encoding="utf-8")
            paths = ExportPaths(prm_steps=prm_path, dpo_pairs=dpo_path, stats_json=stats_path)
        return stats, paths


def format_report(stats: DatasetStats) -> str:
    """Human-readable console report."""
    lines = [
        "=" * 62,
        "PhysEval dataset report",
        "=" * 62,
        f"trajectories          : {stats.total_trajectories} "
        f"({stats.malformed_lines} malformed lines skipped)",
        f"steps                 : {stats.steps_total}",
        f"pass@1                : {100.0 * stats.pass_at_1_rate:.2f}%",
        f"pass@{stats.k_max_turns}                : {100.0 * stats.pass_at_k_rate:.2f}%",
        f"initially failed      : {stats.initially_failed}",
        f"recovered (self-fix)  : {stats.recovered} ({100.0 * stats.recovery_rate:.2f}%)",
        f"avg recovery turns    : {stats.avg_recovery_turns:.2f}",
        f"mean wall time        : {stats.mean_wall_time_s:.2f} s",
        "",
        "most common fatal violations:",
    ]
    if stats.top_fatal_violations:
        for vc in stats.top_fatal_violations:
            lines.append(f"  {vc.count:>6d}  {vc.name}")
    else:
        lines.append("  (none)")
    lines.append("")
    lines.append("per-domain slices:")
    for dom, sl in stats.domain_slices.items():
        lines.append(
            f"  {dom:<10} tasks={sl.tasks:<5d} "
            f"pass@1={100.0 * sl.pass_at_1:6.2f}%  pass@k={100.0 * sl.pass_at_k:6.2f}%"
        )
    lines.append("=" * 62)
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point for the dataset export pipeline."""
    parser = argparse.ArgumentParser(
        prog="physeval-export-dataset",
        description="Export PhysEval trajectories to PRM/DPO training formats.",
    )
    parser.add_argument("--trajectories", "--input", "-i", default="trajectories.jsonl")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(name)s :: %(message)s")
    exporter = DatasetExporter(args.trajectories, args.out_dir)
    try:
        stats, paths = exporter.export(write=not args.report_only)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(format_report(stats))
    if paths is not None:
        print(f"\nwrote:\n  {paths.prm_steps}\n  {paths.dpo_pairs}\n  {paths.stats_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
