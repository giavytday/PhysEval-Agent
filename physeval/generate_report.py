"""Public evaluation reporter for PhysEval-Agent benchmarks.

Consumes ``eval_results.jsonl`` produced by :mod:`physeval.eval_benchmark`
and renders publication-ready artifacts:

* **Markdown performance tables** -- Pass@1 vs. Pass@4 (oracle-guided self-
  correction) broken out across the Grid, Climate, and Kinetics domains.
* **Conservation-drift reduction** --
  :math:`(\\text{Drift}_{\\text{initial}} - \\text{Drift}_{\\text{final}})
  / \\text{Drift}_{\\text{initial}} \\times 100\\%`, reported in aggregate
  and per domain, including improved/worsened task counts.
* **Failure-mode analysis** -- most frequent fatal violation types per domain
  (e.g., Kirchhoff nodal imbalance vs. CFL Courant breaches).
* **High-resolution matplotlib charts** -- ``pass_rates.png`` and
  ``drift_reduction.png`` suitable for technical blogs and documentation.

Matplotlib is optional: without it the Markdown report is still generated and
chart generation is skipped with a warning. Install charts support with::

    pip install 'physeval-agent[report]'
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

from physeval.eval_benchmark import TaskEvalRecord
from physeval.export_dataset import DomainCount

LOGGER = logging.getLogger("physeval.report")


# --------------------------------------------------------------------------- #
# Aggregation                                                                 #
# --------------------------------------------------------------------------- #

class DomainStats(BaseModel):
    """Pass-rate counters for one slice of tasks."""

    tasks: int = 0
    p1_count: int = 0
    pk_count: int = 0

    @property
    def pass_at_1(self) -> float:
        return self.p1_count / self.tasks if self.tasks else 0.0

    @property
    def pass_at_k(self) -> float:
        return self.pk_count / self.tasks if self.tasks else 0.0


class DriftStats(BaseModel):
    """Conservation-drift reduction aggregates."""

    measured: int = 0
    mean_reduction: Optional[float] = None
    improved: int = 0
    unchanged_or_worse: int = 0


class ReportData(BaseModel):
    """Everything needed to render tables and charts."""

    k_turns: int = 4
    title: str = "PhysEval benchmark report"
    total_tasks: int = 0
    malformed_lines: int = 0
    overall: DomainStats = Field(default_factory=DomainStats)
    domains: Dict[str, DomainStats] = Field(default_factory=dict)
    drift: DriftStats = Field(default_factory=DriftStats)
    domain_drift_mean: Dict[str, float] = Field(default_factory=dict)
    domain_drift_counts: Dict[str, int] = Field(default_factory=dict)
    failure_modes: Dict[str, List[DomainCount]] = Field(default_factory=dict)


def parse_records(results_path: str | Path) -> Tuple[List[TaskEvalRecord], int]:
    """Load eval records; returns ``(records, malformed_line_count)``."""
    path = Path(results_path)
    if not path.is_file():
        raise FileNotFoundError(f"Results file not found: {path}")
    records: List[TaskEvalRecord] = []
    malformed = 0
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(TaskEvalRecord.model_validate(json.loads(line)))
            except Exception as exc:  # noqa: BLE001 -- tolerate partial result files
                malformed += 1
                LOGGER.warning("Skipping malformed record at line %d: %s", lineno, exc)
    return records, malformed


def _failure_modes(record: TaskEvalRecord) -> List[str]:
    """Baseline failure labels: oracle violations or exec error class."""
    names = list(record.baseline.final_violation_names)
    if not names and record.baseline.error_kind:
        names = [f"exec:{record.baseline.error_kind}"]
    return names


def build_report(
    records: Sequence[TaskEvalRecord],
    *,
    k_turns: int = 4,
    title: str = "PhysEval benchmark report",
    top_n_modes: int = 5,
) -> ReportData:
    """Aggregate per-task records into a renderable report."""
    report = ReportData(k_turns=k_turns, title=title, total_tasks=len(records))
    mode_counters: Dict[str, Dict[str, int]] = {}
    drift_by_domain: Dict[str, List[float]] = {}
    all_drift: List[float] = []

    for rec in records:
        overall = report.overall
        overall.tasks += 1
        overall.p1_count += int(rec.baseline.success)
        overall.pk_count += int(rec.agentic.success)

        slot = report.domains.setdefault(rec.domain, DomainStats())
        slot.tasks += 1
        slot.p1_count += int(rec.baseline.success)
        slot.pk_count += int(rec.agentic.success)

        for name in _failure_modes(rec):
            mode_counters.setdefault(rec.domain, {})
            mode_counters[rec.domain][name] = mode_counters[rec.domain].get(name, 0) + 1

        if rec.drift_reduction is not None:
            all_drift.append(rec.drift_reduction)
            drift_by_domain.setdefault(rec.domain, []).append(rec.drift_reduction)

    report.malformed_lines = 0  # set by caller via parse_records when relevant

    report.drift.measured = len(all_drift)
    if all_drift:
        report.drift.mean_reduction = sum(all_drift) / len(all_drift)
    report.drift.improved = sum(1 for x in all_drift if x > 0.0)
    report.drift.unchanged_or_worse = sum(1 for x in all_drift if x <= 0.0)

    report.domain_drift_mean = {
        dom: sum(vals) / len(vals) for dom, vals in sorted(drift_by_domain.items()) if vals
    }
    report.domain_drift_counts = {
        dom: len(vals) for dom, vals in sorted(drift_by_domain.items()) if vals
    }
    report.failure_modes = {
        dom: [
            DomainCount(name=name, count=count)
            for name, count in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[:top_n_modes]
        ]
        for dom, counter in sorted(mode_counters.items())
    }
    return report


# --------------------------------------------------------------------------- #
# Markdown rendering                                                          #
# --------------------------------------------------------------------------- #

def _pct(x: float) -> str:
    return f"{100.0 * x:.2f}%"


def render_markdown(report: ReportData) -> str:
    """Render the full Markdown report."""
    k = report.k_turns
    lines: List[str] = [
        f"# {report.title}",
        "",
        f"*Tasks evaluated:* **{report.total_tasks}** "
        f"({report.malformed_lines} malformed result lines skipped)",
        "",
        "## Headline results",
        "",
        "| Scope | Tasks | Pass@1 | "
        f"Pass@{k} (Oracle Self-Correction) |",
        "|---|---:|---:|---:|",
    ]
    for dom, st in report.domains.items():
        lines.append(f"| `{dom}` | {st.tasks} | {_pct(st.pass_at_1)} | {_pct(st.pass_at_k)} |")
    ov = report.overall
    lines.append(
        f"| **Overall** | **{ov.tasks}** | **{_pct(ov.pass_at_1)}** | "
        f"**{_pct(ov.pass_at_k)}** |"
    )

    lines += [
        "",
        f"`Pass@{k}` uses the full Generate → Execute → Verify → Correct loop "
        f"(max {k} turns) where every failed attempt receives the structured "
        f"physics-oracle diagnostic payload.",
        "",
        "## Conservation-drift reduction",
        "",
        r"$$\text{Reduction} = \frac{\text{Drift}_{\text{initial}} - "
        r"\text{Drift}_{\text{final}}}{\text{Drift}_{\text{initial}}} \times 100\%$$",
        "",
    ]
    d = report.drift
    if d.measured == 0:
        lines.append("*No conservation-violating trajectories were measured.*")
    else:
        mean_pct = f"{100.0 * (d.mean_reduction or 0.0):+.2f}%"
        lines += [
            f"- Measured on **{d.measured}** task(s) whose baseline violated a "
            "conservation invariant.",
            f"- Mean relative reduction after oracle-guided correction: **{mean_pct}**",
            f"- Improved: {d.improved} · unchanged/worsened: {d.unchanged_or_worse}",
            "",
            "| Domain | Measured | Mean reduction |",
            "|---|---:|---:|",
        ]
        for dom, val in report.domain_drift_mean.items():
            measured = report.domain_drift_counts.get(dom, 0)
            lines.append(f"| `{dom}` | {measured} | {100.0 * val:+.2f}% |")
        lines.append("")

    lines += ["## Failure modes by domain", ""]
    if not report.failure_modes:
        lines.append("*All baseline attempts passed verification.*")
    else:
        lines += ["| Domain | Failure mode | Occurrences |", "|---|---|---:|"]
        for dom, modes in report.failure_modes.items():
            for i, mc in enumerate(modes):
                label = f"`{dom}`" if i == 0 else ""
                lines.append(f"| {label} | `{mc.name}` | {mc.count} |")
        lines += [
            "",
            "*Reading guide:* `nodal_power_balance` denotes Kirchhoff nodal "
            "imbalance beyond 1e-4 MW; `cfl_stability` denotes Courant numbers "
            "above 1.0; `exec:*` rows are sandbox failures (syntax, imports, "
            "timeouts) rather than physics violations.",
        ]

    lines += [
        "",
        "## Methodology",
        "",
        "- Identical base model, temperature 0, and sandbox budgets across modes.",
        "- Baseline = single-shot generation (max 1 turn); agentic = oracle-in-",
        "  the-loop repair up to the stated turn budget.",
        "- Conservation observables are selected deterministically per task:",
        "  tracer mass drift (climate), nodal power imbalance (grid), or",
        "  cyclic steady-state drift / capture residual (kinetics).",
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Charts                                                                      #
# --------------------------------------------------------------------------- #

def make_charts(report: ReportData, out_dir: Path, *, dpi: int = 200) -> List[Path]:
    """Render high-resolution PNG charts; returns paths actually written."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        LOGGER.warning(
            "matplotlib unavailable; skipping charts. "
            "Install with: pip install 'physeval-agent[report]'"
        )
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []

    # ---- Chart 1: pass rates ------------------------------------------- #
    slices = [(dom, st) for dom, st in report.domains.items()]
    labels = [dom for dom, _ in slices] + ["overall"]
    p1_vals = [st.pass_at_1 * 100.0 for _, st in slices] + [report.overall.pass_at_1 * 100.0]
    pk_vals = [st.pass_at_k * 100.0 for _, st in slices] + [report.overall.pass_at_k * 100.0]

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    x = range(len(labels))
    width = 0.36
    bars1 = ax.bar([i - width / 2 for i in x], p1_vals, width,
                   label="Pass@1 (no feedback)", color="#4C72B0")
    bars2 = ax.bar([i + width / 2 for i in x], pk_vals, width,
                   label=f"Pass@{report.k_turns} (oracle self-correction)",
                   color="#55A868")
    for bars in (bars1, bars2):
        for bar in bars:
            ax.annotate(f"{bar.get_height():.1f}%",
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("pass rate (%)")
    ax.set_ylim(0, 108)
    ax.set_title(report.title)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path1 = out_dir / "pass_rates.png"
    fig.savefig(path1, dpi=dpi)
    plt.close(fig)
    written.append(path1)

    # ---- Chart 2: drift reduction --------------------------------------- #
    if report.domain_drift_mean or report.drift.mean_reduction is not None:
        fig, ax = plt.subplots(figsize=(7.0, 4.2))
        doms = list(report.domain_drift_mean.keys())
        vals = [100.0 * report.domain_drift_mean[d] for d in doms]
        colors = ["#55A868" if v >= 0 else "#C44E52" for v in vals]
        ax.bar(doms, vals, color=colors, alpha=0.9)
        if report.drift.mean_reduction is not None:
            ax.axhline(100.0 * report.drift.mean_reduction, color="#333333",
                       linestyle="--", linewidth=1.2,
                       label=f"overall mean {100.0 * report.drift.mean_reduction:+.1f}%")
            ax.legend(fontsize=9)
        ax.set_ylabel("conservation-drift reduction (%)")
        ax.set_title("Oracle-guided correction: violation severity reduction")
        ax.grid(axis="y", alpha=0.3)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        fig.tight_layout()
        path2 = out_dir / "drift_reduction.png"
        fig.savefig(path2, dpi=dpi)
        plt.close(fig)
        written.append(path2)

    return written


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #

def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="physeval-generate-report",
        description="Render markdown tables + charts from PhysEval evaluation results.",
    )
    parser.add_argument("--results", "--input", "-i", default="eval_results.jsonl")
    parser.add_argument("--out-dir", default="reports")
    parser.add_argument("--title", default="PhysEval benchmark report")
    parser.add_argument("--k", type=int, default=4, help="Agentic turn budget label.")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--no-charts", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point; returns a process exit code."""
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)-7s %(name)s :: %(message)s")

    try:
        records, malformed = parse_records(args.results)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not records:
        print("error: no valid evaluation records found.", file=sys.stderr)
        return 2

    report = build_report(records, k_turns=max(2, args.k), title=args.title)
    report.malformed_lines = malformed

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    markdown = render_markdown(report)
    report_path = out_dir / "report.md"
    report_path.write_text(markdown, encoding="utf-8")

    chart_paths: List[Path] = []
    if not args.no_charts:
        chart_paths = make_charts(report, out_dir, dpi=args.dpi)

    print(markdown)
    print("\nartifacts:")
    print(f"  {report_path}")
    for cp in chart_paths:
        print(f"  {cp}")
    if not args.no_charts and not chart_paths:
        print("  (charts skipped: matplotlib not installed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
