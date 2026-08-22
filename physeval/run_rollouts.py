"""High-throughput asynchronous batch rollout engine.

Runs PhysEval-Agent rollouts for one or more seed tasks in parallel against
any OpenAI-compatible chat-completions endpoint, and exports every multi-turn
trajectory to a dense JSONL file suitable for Process Reward Model (PRM)
training and supervised fine-tuning (SFT).

Usage:
    export OPENAI_API_KEY=sk-...
    python -m physeval.run_rollouts --tasks all --model gpt-4o-mini \
        --output runs/trajectories.jsonl --concurrency 4

Each JSONL line is a full trajectory: per-turn prompts, raw model responses,
extracted code, unified diffs, sandbox stdout/stderr, and structured oracle
verdicts, plus compact ``prm_steps`` reward signals.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, List, Optional

# Allow `python physeval/run_rollouts.py` from the repository root.
_PKG_PARENT = Path(__file__).resolve().parent.parent
if str(_PKG_PARENT) not in sys.path:
    sys.path.insert(0, str(_PKG_PARENT))

from physeval.agent.loop import AsyncPhysEvalLoop, LLMError, RolloutTask, Trajectory  # noqa: E402
from physeval.sandbox.executor import CodeExecutor  # noqa: E402
from physeval.tasks.seed_tasks import all_seed_tasks  # noqa: E402

LOGGER = logging.getLogger("physeval.rollouts")


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="physeval-rollouts",
        description="Async batch rollout engine with JSONL trajectory logging.",
    )
    parser.add_argument(
        "--tasks",
        default="all",
        help="Comma-separated task ids or 'all' (default: all).",
    )
    parser.add_argument(
        "-o", "--output",
        default="trajectories.jsonl",
        help="Output JSONL path (default: trajectories.jsonl).",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        help="Chat model id (default: OPENAI_MODEL env or gpt-4o-mini).",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="OpenAI-compatible API base URL (e.g. http://localhost:8000/v1).",
    )
    parser.add_argument(
        "--suite",
        default=None,
        help="Path to a synthesized benchmark_suite.jsonl (overrides --tasks).",
    )
    parser.add_argument(
        "--client",
        default="openai",
        choices=["openai", "mock"],
        help="'openai' uses the OpenAI-compatible API; 'mock' is a deterministic "
             "offline stand-in for smoke tests.",
    )
    parser.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        help="Environment variable holding the API key (default: OPENAI_API_KEY).",
    )
    parser.add_argument("--max-turns", type=int, default=4, help="Rollout budget (default: 4).")
    parser.add_argument("--concurrency", type=int, default=4, help="Parallel rollouts (default: 4).")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--timeout-s", type=float, default=20.0, help="Sandbox wall-clock budget.")
    parser.add_argument("--mem-limit-mb", type=int, default=None, help="Sandbox memory cap (MiB).")
    parser.add_argument(
        "--with-skeleton",
        action="store_true",
        help="Append each task's starter-code skeleton to its description.",
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="Print the available task ids and exit.",
    )
    return parser.parse_args(argv)


def _resolve_tasks(spec: str) -> List[RolloutTask]:
    """Expand the ``--tasks`` argument into concrete benchmark tasks."""
    spec = spec.strip().lower()
    if spec in {"all", "*"}:
        return all_seed_tasks()
    ids = [s.strip() for s in spec.split(",") if s.strip()]
    return all_seed_tasks(ids)


def _resolve_tasks_from_suite(suite_path: str) -> List[RolloutTask]:
    """Load tasks from a synthesized ``benchmark_suite.jsonl`` file."""
    from physeval.tasks.synthesizer import PhysicsTaskSynthesizer, SynthTaskSpec

    specs = [
        SynthTaskSpec.model_validate(json.loads(line))
        for line in Path(suite_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    synth = PhysicsTaskSynthesizer()
    return [synth.spec_to_task(s) for s in specs]


def _build_client(args: argparse.Namespace) -> Optional[Any]:
    """Instantiate the chat backend: deterministic mock or OpenAI-compatible."""
    if args.client == "mock":
        from physeval.mock_client import MockChatClient

        LOGGER.info("Using deterministic mock client (offline smoke mode).")
        return MockChatClient()
    try:
        from openai import AsyncOpenAI
    except ImportError:
        LOGGER.error(
            "The 'openai' package is required for rollouts. "
            "Install with: pip install 'physeval-agent[llm]'"
        )
        return None
    api_key = os.environ.get(args.api_key_env) or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        LOGGER.error("No API key found in environment variable %s.", args.api_key_env)
        return None
    return AsyncOpenAI(api_key=api_key, base_url=args.base_url)


def _maybe_append_skeleton(task: RolloutTask) -> RolloutTask:
    """Attach starter code to the description when available."""
    starter = getattr(task, "starter_code", None)
    if not starter:
        return task
    suffix = f"\n\n## Starter skeleton\n```python\n{starter}\n```\n"
    return task.model_copy(update={"description": task.description + suffix})


async def run_batch(args: argparse.Namespace) -> int:
    """Execute the configured batch; returns a process exit code."""
    client = _build_client(args)
    if client is None:
        return 2

    executor = CodeExecutor(timeout_s=args.timeout_s, mem_limit_mb=args.mem_limit_mb)
    rollout_loop = AsyncPhysEvalLoop(
        client,
        model=args.model,
        executor=executor,
        max_turns=max(1, args.max_turns),
        temperature=args.temperature,
    )

    try:
        if getattr(args, "suite", None):
            tasks = _resolve_tasks_from_suite(args.suite)
        else:
            tasks = _resolve_tasks(args.tasks)
    except (KeyError, FileNotFoundError) as exc:
        LOGGER.error("%s", exc)
        return 2
    if args.with_skeleton:
        tasks = [_maybe_append_skeleton(t) for t in tasks]
    if not tasks:
        LOGGER.error("No tasks resolved from --tasks=%r.", args.tasks)
        return 2

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_lock = asyncio.Lock()
    stats = {"success": 0, "failure": 0}

    async def worker(task: RolloutTask) -> None:
        LOGGER.info("Starting rollout for %s ...", task.id)
        trajectory: Optional[Trajectory] = None
        try:
            trajectory = await rollout_loop.run(task)
        except LLMError as exc:
            LOGGER.error("[%s] LLM backend failure: %s", task.id, exc)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            LOGGER.exception("[%s] unexpected rollout error: %s", task.id, exc)

        if trajectory is None:
            stats["failure"] += 1
            return

        record = trajectory.to_jsonl_dict()
        async with write_lock:
            with output_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        verdict = "PASS" if trajectory.success else "FAIL"
        LOGGER.info(
            "[%s] %s after %d turn(s), %.1fs",
            task.id, verdict, trajectory.turns_used, trajectory.wall_time_s,
        )
        stats["success" if trajectory.success else "failure"] += 1

    await asyncio.gather(*(worker(t) for t in tasks))
    executor.cleanup()

    total = stats["success"] + stats["failure"]
    print(f"\nBatch complete: {stats['success']}/{total} succeeded "
          f"| trajectories -> {output_path}")
    return 0 if stats["failure"] == 0 and total > 0 else 1


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
        return asyncio.run(run_batch(args))
    except KeyboardInterrupt:
        print("\nInterrupted; partial trajectories already written remain valid JSONL.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
