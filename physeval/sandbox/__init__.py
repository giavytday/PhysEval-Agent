"""Sandbox subpackage: isolated subprocess execution of model code."""

from __future__ import annotations

from physeval.sandbox.executor import (
    Artifact,
    CodeExecutor,
    ErrorKind,
    ExecutionResult,
)

__all__ = ["Artifact", "CodeExecutor", "ErrorKind", "ExecutionResult"]
