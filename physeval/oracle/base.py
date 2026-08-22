"""Abstract verifier schemas shared by every PhysEval physics oracle.

This module defines:

* :class:`InvariantViolation` -- one broken physical invariant.
* :class:`VerificationResult` -- aggregated outcome of an oracle run.
* :class:`BasePhysicsOracle`  -- abstract base class all oracles subclass.
* :class:`StateFileError`     -- raised when a state artifact is unreadable.
"""

from __future__ import annotations

import math
import sys
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Dict, List, Literal, Mapping, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

#: Severity levels. ``FATAL`` fails verification; ``WARNING`` does not.
Severity = Literal["FATAL", "WARNING"]

_MAX_FLOAT = sys.float_info.max


def _finite_or_clamp(value: Any) -> float:
    """Coerce *value* to a finite float.

    NaN collapses to ``0.0``; +/- infinity saturates to the largest finite
    float magnitude so that results remain JSON-serializable.
    """
    out = float(value)
    if math.isnan(out):
        return 0.0
    if math.isinf(out):
        return math.copysign(_MAX_FLOAT, out)
    return out


class InvariantViolation(BaseModel):
    """A single violated physical invariant.

    Attributes:
        name: Stable machine-readable identifier, e.g. ``nodal_power_balance``
            or ``cfl_stability``. Used as the key when feeding diagnostics
            back to the repairing agent.
        severity: ``FATAL`` fails the rollout, ``WARNING`` is advisory only.
        observed_value: The measured quantity that breached the invariant.
        threshold: The limit the observed value was compared against.
        message: Human-readable explanation including units and context
            (bus id, snapshot index, variable name, ...).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(..., min_length=1)
    severity: Severity = "FATAL"
    observed_value: float
    threshold: float
    message: str

    @field_validator("observed_value", "threshold")
    @classmethod
    def _finite(cls, v: float) -> float:
        return _finite_or_clamp(v)


class VerificationResult(BaseModel):
    """Aggregated outcome of an oracle verification pass.

    Attributes:
        passed: True iff no ``FATAL`` violation was recorded.
        violations: All violations found (fatal and warning).
        metrics: Scalar observables describing the state (e.g.
            ``max_nodal_imbalance_mw``, ``max_courant_number``).
    """

    model_config = ConfigDict(extra="forbid")

    passed: bool
    violations: List[InvariantViolation] = Field(default_factory=list)
    metrics: Dict[str, float] = Field(default_factory=dict)

    @field_validator("metrics")
    @classmethod
    def _sanitize_metrics(cls, v: Dict[str, float]) -> Dict[str, float]:
        return {str(k): _finite_or_clamp(x) for k, x in v.items()}

    @model_validator(mode="after")
    def _sync_passed_with_violations(self) -> VerificationResult:
        has_fatal = any(v.severity == "FATAL" for v in self.violations)
        if self.passed and has_fatal:
            raise ValueError(
                "`passed=True` contradicts FATAL violations present in `violations`."
            )
        return self

    @property
    def fatal_violations(self) -> List[InvariantViolation]:
        """Violations that caused failure."""
        return [v for v in self.violations if v.severity == "FATAL"]

    @property
    def warnings(self) -> List[InvariantViolation]:
        """Advisory violations that did not cause failure."""
        return [v for v in self.violations if v.severity == "WARNING"]

    @classmethod
    def ok(cls, metrics: Optional[Mapping[str, float]] = None) -> VerificationResult:
        """Build a passing result."""
        return cls(passed=True, violations=[], metrics=dict(metrics or {}))

    @classmethod
    def failed(
        cls,
        violations: List[InvariantViolation],
        metrics: Optional[Mapping[str, float]] = None,
    ) -> VerificationResult:
        """Build a failing result from at least one FATAL violation."""
        if not any(v.severity == "FATAL" for v in violations):
            raise ValueError("`failed()` requires at least one FATAL violation.")
        return cls(passed=False, violations=violations, metrics=dict(metrics or {}))

    def summarize(self) -> str:
        """Compact multi-line human summary used in LLM repair prompts."""
        lines = ["PASSED" if self.passed else "FAILED"]
        for v in self.violations:
            lines.append(
                f"[{v.severity}] {v.name}: observed={v.observed_value:.6g} "
                f"threshold={v.threshold:.6g} :: {v.message}"
            )
        for k, m in self.metrics.items():
            lines.append(f"metric {k}={m:.6g}")
        return "\n".join(lines)


class StateFileError(RuntimeError):
    """Raised when a serialized state file cannot be located or parsed."""


class BasePhysicsOracle(ABC):
    """Abstract contract for deterministic physics verifiers.

    Subclasses must implement :meth:`verify`, which loads a state artifact
    written by untrusted sandbox code and checks physical invariants. Oracles
    must be pure functions of the state file: no randomness, no network I/O,
    no dependence on ambient interpreter state, so identical artifacts always
    yield identical verdicts.
    """

    #: Stable identifier used in trajectories and logs.
    name: ClassVar[str] = "base-physics-oracle"

    #: File suffixes this oracle can consume.
    supported_extensions: ClassVar[tuple] = ()

    @abstractmethod
    def verify(self, state_file_path: str) -> VerificationResult:
        """Load the state artifact at *state_file_path* and check invariants.

        Implementations should raise :class:`StateFileError` only for missing
        or unparsable files; broken physics must be reported through
        FATAL violations instead of exceptions so the agent loop receives
        structured diagnostics.

        Args:
            state_file_path: Path to the exported state (e.g. ``network.nc``).

        Returns:
            A fully populated :class:`VerificationResult`.
        """

    def validate_path(self, state_file_path: str) -> None:
        """Shared pre-checks: existence, readability, extension match.

        Raises:
            StateFileError: If the path is missing/unreadable or has an
                unsupported extension.
        """
        import os

        if not os.path.isfile(state_file_path):
            raise StateFileError(f"State file does not exist: {state_file_path!r}")
        if not os.access(state_file_path, os.R_OK):
            raise StateFileError(f"State file is not readable: {state_file_path!r}")
        ext = os.path.splitext(state_file_path)[1].lower()
        if self.supported_extensions and ext not in self.supported_extensions:
            raise StateFileError(
                f"{self.name} supports extensions {self.supported_extensions}, "
                f"got {ext!r} ({state_file_path!r})"
            )

    @staticmethod
    def make_violation(
        name: str,
        severity: Severity,
        observed: float,
        threshold: float,
        message: str,
    ) -> InvariantViolation:
        """Factory helper so subclasses build uniformly-shaped violations."""
        return InvariantViolation(
            name=name,
            severity=severity,
            observed_value=observed,
            threshold=threshold,
            message=message,
        )
