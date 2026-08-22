"""Physics oracle subpackage.

Oracles are deterministic verifiers that load a serialized simulation state
(NetCDF / HDF5 / Zarr) produced inside the sandbox and check physical
invariants (conservation laws, bounds, stability criteria). Every oracle
returns a structured :class:`~physeval.oracle.base.VerificationResult`.
"""

from __future__ import annotations

from physeval.oracle.base import (
    BasePhysicsOracle,
    InvariantViolation,
    Severity,
    StateFileError,
    VerificationResult,
)

__all__ = [
    "BasePhysicsOracle",
    "InvariantViolation",
    "Severity",
    "StateFileError",
    "VerificationResult",
]
