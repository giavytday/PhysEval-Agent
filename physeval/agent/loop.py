"""Multi-turn rollout orchestrator: Generate -> Execute -> Verify -> Correct.

:class:`AsyncPhysEvalLoop` drives one benchmark task against an
OpenAI-compatible chat-completions client:

1. *Generate*: prompt the model with the task specification and output
   contract; extract the candidate script.
2. *Execute*: run the script inside the :class:`CodeExecutor` sandbox with
   wall-clock and memory budgets.
3. *Verify*: hand exported state artifacts to the task's physics oracle.
4. *Correct*: on execution failure or physical violations, feed a structured
   JSON diagnostic payload (exact imbalances, Courant numbers, stderr) back to
   the model requesting a targeted patch, for up to ``max_turns`` rounds.

Every step is recorded in an immutable :class:`Trajectory` suitable for PRM
training and SFT export.
"""

from __future__ import annotations

import difflib
import logging
import time
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from physeval.agent.prompts import (
    SYSTEM_PROMPT,
    build_generation_prompt,
    build_repair_prompt,
    extract_code,
)
from physeval.oracle.base import (
    BasePhysicsOracle,
    StateFileError,
    VerificationResult,
)
from physeval.sandbox.executor import CodeExecutor, ExecutionResult

__all__ = ["AsyncPhysEvalLoop", "LLMError", "RolloutTask", "Trajectory", "TrajectoryStep"]

LOGGER = logging.getLogger("physeval.agent")

#: Hard cap on any single text field persisted in a step (chars).
_MAX_TEXT_CHARS = 30_000

_FORMAT_NUDGE = (
    "Your previous reply could not be parsed into a Python script. Respond "
    "again with EXACTLY ONE fenced ```python block containing the complete "
    "script (or the strict JSON repair schema if repair instructions were "
    "given). No prose outside the block."
)


class LLMError(RuntimeError):
    """Raised when the chat-completions backend fails after retries."""


class RolloutTask(BaseModel):
    """Specification of one benchmark problem handed to the loop.

    Attributes:
        id: Stable unique identifier, e.g. ``grid_24h_curtailment``.
        title: Short human-readable name.
        description: Full prose statement of the modeling problem.
        requirements: Enumerated physics/IO requirements the oracle checks.
        artifact_filename: Exact relative path of the serialized state the
            generated code must write (e.g. ``network_state.nc``).
        oracle: The deterministic verifier deciding pass/fail.
        tags: Free-form labels used for slicing benchmark results.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str = Field(..., min_length=1)
    title: str
    description: str
    requirements: List[str] = Field(default_factory=list)
    artifact_filename: str
    oracle: BasePhysicsOracle
    tags: List[str] = Field(default_factory=list)


class TrajectoryStep(BaseModel):
    """One generate/repair round-trip with its full evidence bundle."""

    turn: int
    stage: Literal["generate", "repair"]
    prompt: str
    raw_response: str = ""
    extracted_code: Optional[str] = None
    code_diff_unified: Optional[str] = None
    execution: Optional[Dict[str, Any]] = None
    verification: Optional[Dict[str, Any]] = None

    @property
    def executed(self) -> bool:
        return self.execution is not None

    @property
    def exec_ok(self) -> bool:
        return bool(self.execution and self.execution.get("ok"))

    @property
    def oracle_passed(self) -> bool:
        return bool(self.verification and self.verification.get("passed"))

    @property
    def num_fatal_violations(self) -> int:
        if not self.verification:
            return 0
        return sum(
            1
            for v in self.verification.get("violations", [])
            if v.get("severity") == "FATAL"
        )


class Trajectory(BaseModel):
    """Complete multi-turn record for one task rollout."""

    schema_version: str = "1.0"
    task_id: str
    model: str
    max_turns: int
    steps: List[TrajectoryStep] = Field(default_factory=list)
    success: bool = False
    turns_used: int = 0
    wall_time_s: float = 0.0
    final_error_kind: Optional[str] = None

    @property
    def final_step(self) -> Optional[TrajectoryStep]:
        return self.steps[-1] if self.steps else None

    def to_jsonl_dict(self) -> Dict[str, Any]:
        """Dense JSONL projection including per-step PRM reward signals."""
        return {
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "model": self.model,
            "max_turns": self.max_turns,
            "success": self.success,
            "turns_used": self.turns_used,
            "wall_time_s": round(self.wall_time_s, 3),
            "final_error_kind": self.final_error_kind,
            "steps": [s.model_dump() for s in self.steps],
            "prm_steps": [
                {
                    "turn": s.turn,
                    "exec_ok": s.exec_ok,
                    "oracle_passed": s.oracle_passed,
                    "num_fatal_violations": s.num_fatal_violations,
                }
                for s in self.steps
            ],
        }


class AsyncPhysEvalLoop:
    """Asynchronous orchestrator implementing the verify-correct rollout."""

    def __init__(
        self,
        client: Any,
        *,
        model: str = "gpt-4o-mini",
        executor: Optional[CodeExecutor] = None,
        max_turns: int = 4,
        temperature: float = 0.0,
        llm_retries: int = 2,
        llm_backoff_s: float = 1.5,
    ) -> None:
        """Configure the loop.

        Args:
            client: OpenAI-compatible async client exposing
                ``await client.chat.completions.create(model=..., messages=...,
                temperature=...)`` returning an object with
                ``choices[0].message.content``.
            model: Chat model identifier.
            executor: Sandbox instance; a fresh default is created when omitted.
            max_turns: Maximum generate/repair iterations (spec default 4).
            temperature: Sampling temperature; 0.0 keeps rollouts reproducible.
            llm_retries: Attempts per chat call before raising :class:`LLMError`.
            llm_backoff_s: Base delay for exponential retry backoff.
        """
        if max_turns < 1:
            raise ValueError("max_turns must be >= 1.")
        self.client = client
        self.model = str(model)
        self.executor = executor or CodeExecutor()
        self.max_turns = int(max_turns)
        self.temperature = float(temperature)
        self.llm_retries = max(1, int(llm_retries))
        self.llm_backoff_s = float(llm_backoff_s)

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #

    async def run(self, task: RolloutTask) -> Trajectory:
        """Execute the full rollout for *task* and return its trajectory."""
        started = time.monotonic()
        generation_prompt = build_generation_prompt(
            task_id=task.id,
            title=task.title,
            description=task.description,
            requirements=task.requirements,
            artifact_filename=task.artifact_filename,
        )
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": generation_prompt},
        ]

        steps: List[TrajectoryStep] = []
        success = False
        previous_code: Optional[str] = None

        for turn in range(1, self.max_turns + 1):
            stage: Literal["generate", "repair"] = "generate" if turn == 1 else "repair"
            try:
                raw_response = await self._chat(messages)
            except LLMError as exc:
                # Backend outage mid-rollout: record evidence, end gracefully.
                LOGGER.error("[%s] turn %d: LLM backend failure: %s", task.id, turn, exc)
                steps.append(
                    TrajectoryStep(
                        turn=turn,
                        stage=stage,
                        prompt=messages[-1]["content"],
                        raw_response=f"<llm_error>{exc}",
                        extracted_code=None,
                    )
                )
                break
            raw_response = raw_response or ""

            code = extract_code(raw_response)
            if code is None:
                steps.append(
                    TrajectoryStep(
                        turn=turn,
                        stage=stage,
                        prompt=messages[-1]["content"],
                        raw_response=raw_response[:_MAX_TEXT_CHARS],
                        extracted_code=None,
                    )
                )
                messages.append({"role": "assistant", "content": raw_response})
                messages.append({"role": "user", "content": _FORMAT_NUDGE})
                LOGGER.warning("[%s] turn %d: unparseable response", task.id, turn)
                continue

            execution = await self.executor.execute_async(code, run_id=f"{task.id}-t{turn}")
            verification = self._verify(execution, task.oracle)

            steps.append(
                TrajectoryStep(
                    turn=turn,
                    stage=stage,
                    prompt=messages[-1]["content"],
                    raw_response=raw_response[:_MAX_TEXT_CHARS],
                    extracted_code=code[:_MAX_TEXT_CHARS],
                    code_diff_unified=_unified_diff(previous_code, code),
                    execution=execution.to_dict(),
                    verification=verification.model_dump() if verification else None,
                )
            )

            passed = bool(execution.ok and verification is not None and verification.passed)
            LOGGER.info(
                "[%s] turn %d: exec_ok=%s verified=%s",
                task.id,
                turn,
                execution.ok,
                None if verification is None else verification.passed,
            )
            if passed:
                success = True
                break

            messages.append(
                {"role": "assistant", "content": f"```python\n{code}\n```"}
            )
            messages.append(
                {
                    "role": "user",
                    "content": build_repair_prompt(
                        turn=turn,
                        execution=execution.to_dict(),
                        verification=verification.model_dump() if verification else None,
                    ),
                }
            )
            previous_code = code

        last_exec_error = next(
            (
                (s.execution or {}).get("error_kind")
                for s in reversed(steps)
                if s.execution is not None
            ),
            None,
        )
        return Trajectory(
            task_id=task.id,
            model=self.model,
            max_turns=self.max_turns,
            steps=steps,
            success=success,
            turns_used=len(steps),
            wall_time_s=time.monotonic() - started,
            final_error_kind=None if success else last_exec_error,
        )

    # ------------------------------------------------------------------ #
    # Internals                                                          #
    # ------------------------------------------------------------------ #

    async def _chat(self, messages: List[Dict[str, str]]) -> str:
        """Call the chat-completions endpoint with bounded retries."""
        import asyncio

        last_exc: Optional[BaseException] = None
        for attempt in range(self.llm_retries):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,  # type: ignore[arg-type]
                    temperature=self.temperature,
                )
                content = response.choices[0].message.content
                return content if isinstance(content, str) else ""
            except Exception as exc:
                last_exc = exc
                delay = self.llm_backoff_s * (2**attempt)
                LOGGER.warning("LLM call failed (attempt %d): %s; retrying in %.1fs",
                               attempt + 1, exc, delay)
                await asyncio.sleep(delay)
        raise LLMError(f"Chat completion failed after {self.llm_retries} attempts: {last_exc}")

    @staticmethod
    def _verify(
        execution: ExecutionResult,
        oracle: BasePhysicsOracle,
    ) -> Optional[VerificationResult]:
        """Run the oracle over the primary state artifact.

        Returns ``None`` when execution itself failed (execution diagnostics
        dominate the repair prompt). Converts unreadable/missing artifacts and
        verifier crashes into FATAL violations so the agent always receives
        structured feedback instead of exceptions.
        """
        if not execution.ok:
            return None

        state_file = execution.primary_state_file
        if state_file is None:
            return VerificationResult.failed(
                [_artifact_missing_violation(execution)],
                metrics={"num_artifacts": float(len(execution.artifacts))},
            )
        try:
            return oracle.verify(state_file)
        except StateFileError as exc:
            return VerificationResult.failed(
                [
                    oracle.make_violation(
                        name="state_file_unreadable",
                        severity="FATAL",
                        observed=0.0,
                        threshold=1.0,
                        message=str(exc),
                    )
                ]
            )
        except Exception as exc:
            return VerificationResult.failed(
                [
                    oracle.make_violation(
                        name="oracle_crashed",
                        severity="FATAL",
                        observed=float("nan"),
                        threshold=float("nan"),
                        message=f"Verifier raised unexpectedly: {type(exc).__name__}: {exc}",
                    )
                ]
            )


def _artifact_missing_violation(execution: ExecutionResult) -> Any:
    from physeval.oracle.base import InvariantViolation

    return InvariantViolation(
        name="state_artifact_missing",
        severity="FATAL",
        observed_value=0.0,
        threshold=1.0,
        message=(
            "Script exited successfully but exported no recognizable state file. "
            f"Artifacts found: {[a.path for a in execution.artifacts] or 'none'}."
        ),
    )


def _unified_diff(previous: Optional[str], current: str, context: int = 3) -> Optional[str]:
    """Render a unified diff between consecutive candidate scripts."""
    if previous is None:
        return None
    diff = difflib.unified_diff(
        previous.splitlines(),
        current.splitlines(),
        fromfile="previous_attempt",
        tofile="new_attempt",
        lineterm="",
        n=context,
    )
    text = "\n".join(diff)
    return text[:_MAX_TEXT_CHARS] if text else None
