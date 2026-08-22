"""End-to-end integration test: Synthesize -> Generate -> Verify -> Export -> Report.

Runs one task per domain (grid / climate / kinetics) through the *real* CLI
entry points of every pipeline stage using the deterministic mock LLM client,
then validates every artifact: JSONL schema conformance, absence of NaN/Infin
constants, reward sanity, pass-rate invariants, and report chart generation.

Runtime: roughly 1-2 minutes (dominated by sandbox subprocess spins).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

from physeval.eval_benchmark import TaskEvalRecord
from physeval.export_dataset import PRMStepRecord
from physeval.generate_report import main as generate_report_main
from physeval.run_rollouts import main as run_rollouts_main

pytest.importorskip("xarray")
pytest.importorskip("netCDF4")


def _reject_constant(name: str) -> float:
    """json.loads(parse_constant=...) hook: NaN/Infinity are hard failures."""
    raise ValueError(f"non-finite JSON constant: {name}")


def _strict_loads(line: str) -> Any:
    return json.loads(line, parse_constant=_reject_constant)


@pytest.fixture(scope="module")
def suite_file(tmp_path_factory) -> Path:
    from physeval.tasks.synthesizer import PhysicsTaskSynthesizer

    suite = tmp_path_factory.mktemp("e2e") / "benchmark_suite.jsonl"
    specs = PhysicsTaskSynthesizer(seed=123).synthesize(per_domain=1)
    PhysicsTaskSynthesizer.export_jsonl(specs, suite)
    assert suite.is_file()
    return suite


@pytest.fixture(scope="module")
def workspace(tmp_path_factory, suite_file) -> Dict[str, Path]:
    """Run rollouts + export once for the whole module."""
    ws = tmp_path_factory.mktemp("pipeline_ws")
    data_dir = ws / "data"
    traj = data_dir / "trajectories.jsonl"

    rc = run_rollouts_main([
        "--suite", str(suite_file),
        "--client", "mock",
        "--concurrency", "3",
        "--max-turns", "3",
        "--output", str(traj),
    ])
    assert rc == 0, "run_rollouts exited non-zero"

    from physeval.export_dataset import main as export_dataset_main

    rc = export_dataset_main(["--trajectories", str(traj), "--out-dir", str(data_dir)])
    assert rc == 0, "export_dataset exited non-zero"

    return {"ws": ws, "suite": suite_file, "traj": traj, "data_dir": data_dir}


def test_rollouts_covers_all_three_domains(workspace):
    traj: Path = workspace["traj"]
    lines = [line for line in traj.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 3, "expected exactly one trajectory per domain task"

    task_ids = {str(_strict_loads(line)["task_id"]) for line in lines}
    assert task_ids == {"grid_synth_0000", "climate_synth_0000", "kinetics_synth_0000"}

    for line in lines:
        record: Dict[str, Any] = _strict_loads(line)  # raises on NaN/Infinity
        assert record["schema_version"] == "1.0"
        assert isinstance(record["success"], bool)
        assert record["steps"], "trajectory must contain at least one step"
        for step in record["steps"]:
            assert {"turn", "stage", "prompt", "raw_response"} <= set(step)
            execution = step.get("execution")
            if execution is not None:
                assert isinstance(execution["ok"], bool)
                assert isinstance(execution["error_kind"], str)


def test_prm_export_schema_and_rewards(workspace):
    prm_path: Path = workspace["data_dir"] / "prm_steps.jsonl"
    assert prm_path.is_file()
    rows = [
        PRMStepRecord.model_validate(_strict_loads(line))
        for line in prm_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    # Broken first attempt + corrected repair for climate and kinetics at
    # minimum; grid contributes steps too (failing when pypsa is absent).
    assert len(rows) >= 6
    rewards = {r.reward for r in rows}
    assert 0.0 in rewards, "mock first attempts must yield negative process labels"
    assert 1.0 in rewards, "repaired climate/kinetics solutions must be rewarded"
    for row in rows:
        if row.code_span_chars is not None:
            span = row.code_span_chars
            assert row.completion[span["start"]:span["end"]] == row.code


def test_dpo_pairs_and_stats(workspace):
    data_dir: Path = workspace["data_dir"]
    dpo_path = data_dir / "dpo_pairs.jsonl"
    stats_path = data_dir / "stats.json"
    assert dpo_path.is_file() and stats_path.is_file()

    dpo_lines = [line for line in dpo_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    # Climate + kinetics self-corrections are deterministic; grid only when
    # PyPSA is installed in the sandbox environment.
    assert len(dpo_lines) >= 2
    pairs = [_strict_loads(line) for line in dpo_lines]
    for pair in pairs:
        assert pair["rejected"] != pair["chosen"]
        assert pair["rejected_reward"] == 0.0 and pair["chosen_reward"] == 1.0
        assert pair["turns_to_recover"] >= 1
        assert pair["prompt"].strip(), "pairs must condition on repair context"

    stats = _strict_loads(stats_path.read_text(encoding="utf-8"))
    assert stats["total_trajectories"] == 3
    assert stats["malformed_lines"] == 0
    assert stats["prm_records"] >= len(dpo_lines)
    assert stats["pass_at_k_rate"] >= stats["pass_at_1_rate"]
    assert 0.0 <= stats["pass_at_1_rate"] <= 1.0
    assert stats["avg_recovery_turns"] >= 1.0


def test_benchmark_evaluation_records(workspace):
    eval_results = workspace["ws"] / "eval_results.jsonl"

    from physeval.eval_benchmark import main as eval_benchmark_main

    rc = eval_benchmark_main([
        "--suite", str(workspace["suite"]),
        "--client", "mock",
        "--max-turns", "3",
        "--concurrency", "3",
        "--output", str(eval_results),
    ])
    assert rc == 0, "eval_benchmark exited non-zero"

    records = [
        TaskEvalRecord.model_validate(_strict_loads(line))
        for line in eval_results.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(records) == 3

    # By design every mock first attempt is physically flawed...
    assert all(not r.baseline.success for r in records)
    # ...and oracle-guided self-correction rescues climate + kinetics
    # deterministically (grid additionally when PyPSA is available).
    recovered = sum(1 for r in records if r.agentic.success)
    assert recovered >= 2

    reductions = [r.drift_reduction for r in records if r.drift_reduction is not None]
    assert len(reductions) >= 2
    assert all(x > 0.0 for x in reductions), "repairs must reduce conservation drift"


def test_report_generation_artifacts(workspace, tmp_path):
    eval_results = workspace["ws"] / "eval_results.jsonl"
    reports_dir = tmp_path / "reports"

    rc = generate_report_main([
        "--results", str(eval_results),
        "--out-dir", str(reports_dir),
        "--title", "PhysEval end-to-end smoke",
        "--k", "3",
    ])
    assert rc == 0

    report_md = (reports_dir / "report.md").read_text(encoding="utf-8")
    assert "# PhysEval end-to-end smoke" in report_md
    assert "Pass@4" in report_md or "Pass@3" in report_md
    assert r"\frac{\text{Drift}_{\text{initial}}" in report_md
    assert "## Failure modes by domain" in report_md

    try:
        import matplotlib  # noqa: F401
    except ImportError:
        pytest.skip("matplotlib not installed; charts skipped")
    for png in ("pass_rates.png", "drift_reduction.png"):
        path = reports_dir / png
        assert path.is_file(), f"missing chart {png}"
        assert path.stat().st_size > 1000, f"chart {png} suspiciously small"


def test_shell_pipeline_runner_smoke(tmp_path):
    """The master runner itself executes all five steps hermetically."""
    import os
    import subprocess
    import sys

    repo_root = Path(__file__).resolve().parent.parent
    env = dict(os.environ)
    env.setdefault("PYTHONPATH", str(repo_root))
    # Ensure the runner uses the SAME interpreter as this test session.
    env["PYTHON"] = sys.executable
    proc = subprocess.run(
        ["bash", str(repo_root / "run_pipeline.sh"),
         "--smoke-test",
         "--out-dir", str(tmp_path / "pipe_out")],
        capture_output=True, text=True, cwd=str(repo_root), timeout=900, env=env,
    )
    assert proc.returncode == 0, f"runner failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
    assert "PIPELINE COMPLETE" in proc.stdout

    out_root = tmp_path / "pipe_out"
    expected = [
        out_root / "tasks" / "benchmark_suite.jsonl",
        out_root / "data" / "trajectories.jsonl",
        out_root / "data" / "prm_steps.jsonl",
        out_root / "data" / "dpo_pairs.jsonl",
        out_root / "data" / "stats.json",
        out_root / "eval_results.jsonl",
        out_root / "reports" / "report.md",
    ]
    for artifact in expected:
        assert artifact.is_file(), f"missing pipeline artifact: {artifact}"
        content = artifact.read_text(encoding="utf-8")
        if artifact.suffix == ".json":  # whole-document JSON (e.g. stats.json)
            _strict_loads(content)
        else:  # JSONL
            for line in content.splitlines():
                if line.strip().startswith("{"):
                    _strict_loads(line)  # no NaN/Infinity anywhere

    summary = json.loads((out_root / "data" / "stats.json").read_text(encoding="utf-8"))
    assert summary["total_trajectories"] == 3
