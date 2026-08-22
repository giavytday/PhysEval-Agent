"""PhysEval-Agent: deterministic scientific execution & verification framework.

Modules:
    physeval.oracle   -- deterministic physics verifiers (grid / climate).
    physeval.sandbox  -- subprocess execution sandbox with timeouts.
    physeval.agent    -- multi-turn rollout orchestration and prompts.
    physeval.tasks    -- seed benchmark suite.
    physeval.run_rollouts -- asynchronous batch rollout CLI.
"""

from __future__ import annotations

__version__ = "0.1.0"

from physeval.agent.loop import (
    AsyncPhysEvalLoop,
    LLMError,
    RolloutTask,
    Trajectory,
    TrajectoryStep,
)
from physeval.oracle.base import (
    BasePhysicsOracle,
    InvariantViolation,
    Severity,
    StateFileError,
    VerificationResult,
)
from physeval.sandbox.executor import (
    Artifact,
    CodeExecutor,
    ErrorKind,
    ExecutionResult,
)

__all__ = [
    "Artifact",
    "AsyncPhysEvalLoop",
    "BasePhysicsOracle",
    "CodeExecutor",
    "ErrorKind",
    "ExecutionResult",
    "InvariantViolation",
    "LLMError",
    "RolloutTask",
    "Severity",
    "StateFileError",
    "Trajectory",
    "TrajectoryStep",
    "VerificationResult",
    "__version__",
]
