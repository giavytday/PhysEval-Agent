"""Benchmark task suite for PhysEval-Agent."""

from __future__ import annotations

from physeval.tasks.seed_tasks import (
    SEED_TASKS,
    BenchmarkTask,
    DACKineticsOracle,
    all_seed_tasks,
    get_seed_task,
)

__all__ = [
    "SEED_TASKS",
    "BenchmarkTask",
    "DACKineticsOracle",
    "all_seed_tasks",
    "get_seed_task",
]
