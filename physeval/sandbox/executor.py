"""Subprocess sandbox for executing untrusted simulation code.

:class:`CodeExecutor` runs raw Python source in a fresh interpreter with:

* a per-run scratch working directory,
* an isolated-mode interpreter (``python -I``) and a minimal, hash-seeded
  environment for reproducibility,
* wall-clock timeout enforcement with whole-process-group kill on expiry,
* an optional address-space cap (POSIX ``setrlimit(RLIMIT_AS)``, applied by a
  trusted in-child bootstrap so the parent never needs ``preexec_fn``),
* stdout/stderr capture with bounded memory footprint,
* automatic discovery of exported state artifacts (``*.nc``, ``*.h5``,
  ``*.zarr``, ...).

Failure modes -- syntax errors, missing imports, timeouts, memory blowups and
generic runtime errors -- are classified into :class:`ErrorKind` values so the
agent loop can craft targeted repair feedback instead of crashing.
"""

from __future__ import annotations

import asyncio
import contextlib
import enum
import os
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = ["Artifact", "CodeExecutor", "ErrorKind", "ExecutionResult"]

#: Default glob patterns used to discover exported state artifacts.
DEFAULT_ARTIFACT_GLOBS: Tuple[str, ...] = (
    "*.nc",
    "*.nc4",
    "*.cdf",
    "*.h5",
    "*.hdf5",
    "*.zarr",
    "*.npz",
    "*.json",
    "*.csv",
)

#: Environment variables pinned inside the child process.
_SANDBOX_ENV: Dict[str, str] = {
    "PYTHONHASHSEED": "0",  # deterministic string hashing
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONUNBUFFERED": "1",
    "MPLBACKEND": "Agg",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "LANG": "C.UTF-8",
}

#: Trusted in-child bootstrap. Applies POSIX rlimits *inside* the executed
#: interpreter (never via ``preexec_fn``, which is unsafe after fork in
#: multithreaded parents) and only then runs the untrusted script.
_SANDBOX_BOOTSTRAP = """\
import resource
import runpy
import sys

mem_bytes, cpu_seconds, script = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    if int(mem_bytes) > 0:
        limit = int(mem_bytes)
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    if int(cpu_seconds) > 0:
        soft, hard = resource.getrlimit(resource.RLIMIT_CPU)
        cap = int(cpu_seconds)
        resource.setrlimit(
            resource.RLIMIT_CPU,
            (min(soft, cap) if soft > 0 else cap,
             min(hard, cap) if hard > 0 else cap),
        )
except Exception:
    pass  # best effort; platforms differ (e.g. macOS RLIMIT_AS quirks)

sys.argv = [script]
runpy.run_path(script, run_name="__main__")
"""

#: Cap on captured stream size (characters); head+tail are preserved.
_STREAM_CAPTURE_LIMIT = 100_000


class ErrorKind(str, enum.Enum):
    """Classification of how a sandboxed execution terminated."""

    NONE = "none"
    SYNTAX = "syntax_error"
    IMPORT = "import_error"
    TIMEOUT = "timeout"
    MEMORY = "memory_error"
    RUNTIME = "runtime_error"


def _truncate(text: str, limit: int = _STREAM_CAPTURE_LIMIT) -> str:
    """Keep the head and tail of oversized streams with an elision marker."""
    if len(text) <= limit:
        return text
    half = limit // 2
    marker = f"\n... [truncated {len(text) - limit} characters] ...\n"
    return text[:half] + marker + text[-half:]


def _last_traceback_line(stderr: str) -> Optional[str]:
    """Return the most informative trailing line of a traceback, if any."""
    lines = [ln.strip() for ln in stderr.strip().splitlines() if ln.strip()]
    return lines[-1] if lines else None


@dataclass(frozen=True)
class Artifact:
    """A state file exported by the sandboxed script."""

    path: str
    kind: str  # e.g. "netcdf", "hdf5", "zarr", "json"

    def to_dict(self) -> Dict[str, str]:
        return {"path": self.path, "kind": self.kind}


@dataclass
class ExecutionResult:
    """Everything observed from one sandbox run."""

    ok: bool
    error_kind: ErrorKind = ErrorKind.NONE
    returncode: Optional[int] = None
    stdout: str = ""
    stderr: str = ""
    duration_s: float = 0.0
    timed_out: bool = False
    workdir: str = ""
    artifacts: List[Artifact] = field(default_factory=list)
    error_summary: Optional[str] = None

    @property
    def primary_state_file(self) -> Optional[str]:
        """Best-guess serialized simulation state for oracle verification."""
        priority = ("*.nc", "*.nc4", "*.cdf", "*.h5", "*.hdf5", "*.zarr")
        for pattern in priority:
            for art in self.artifacts:
                if art.path.endswith(pattern.lstrip("*")):
                    return art.path
        return self.artifacts[0].path if self.artifacts else None

    def to_dict(self) -> Dict[str, Any]:
        """JSON-ready projection for trajectory logging."""
        return {
            "ok": self.ok,
            "error_kind": self.error_kind.value,
            "returncode": self.returncode,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "duration_s": round(self.duration_s, 4),
            "timed_out": self.timed_out,
            "workdir": self.workdir,
            "artifacts": [a.to_dict() for a in self.artifacts],
            "error_summary": self.error_summary,
        }


class CodeExecutor:
    """Deterministic, resource-bounded subprocess runner for model code."""

    def __init__(
        self,
        *,
        timeout_s: float = 20.0,
        mem_limit_mb: Optional[int] = None,
        base_dir: Optional[str] = None,
        artifact_globs: Sequence[str] = DEFAULT_ARTIFACT_GLOBS,
        keep_workdirs: bool = False,
        python_executable: Optional[str] = None,
    ) -> None:
        """Configure the sandbox.

        Args:
            timeout_s: Hard wall-clock budget per run (default 20 s).
            mem_limit_mb: Optional child address-space cap in MiB (POSIX).
            base_dir: Root directory under which scratch dirs are created;
                defaults to the system temp directory.
            artifact_globs: Patterns scanned in the workdir after each run.
            keep_workdirs: Retain scratch directories for post-mortem debug.
            python_executable: Interpreter override; defaults to the running one.
        """
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive.")
        self.timeout_s = float(timeout_s)
        self.mem_limit_mb = mem_limit_mb
        self.base_dir = Path(base_dir) if base_dir else Path(tempfile.gettempdir()) / "physeval_runs"
        self.artifact_globs = tuple(artifact_globs)
        self.keep_workdirs = keep_workdirs
        self.python_executable = python_executable or sys.executable
        self._managed_dirs: List[Path] = []
        self.base_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        with contextlib.suppress(OSError):
            os.chmod(self.base_dir, 0o700)

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #

    def execute(self, code: str, *, run_id: Optional[str] = None) -> ExecutionResult:
        """Execute *code* synchronously and return an :class:`ExecutionResult`."""
        rid = run_id or uuid.uuid4().hex[:12]
        workdir = self.base_dir / f"run_{rid}"
        workdir.mkdir(parents=True, exist_ok=True)
        with contextlib.suppress(OSError):
            os.chmod(workdir, 0o700)
        self._managed_dirs.append(workdir)

        # Preflight: catch syntax errors deterministically before spawning.
        try:
            compile(code, filename="<sandbox_script>", mode="exec")
        except SyntaxError as exc:
            return ExecutionResult(
                ok=False,
                error_kind=ErrorKind.SYNTAX,
                returncode=None,
                stdout="",
                stderr=f"SyntaxError: {exc.msg} (line {exc.lineno}, offset {exc.offset})",
                duration_s=0.0,
                timed_out=False,
                workdir=str(workdir),
                artifacts=[],
                error_summary=f"SyntaxError: {exc.msg} (line {exc.lineno})",
            )

        script_path = workdir / "script.py"
        script_path.write_text(code, encoding="utf-8")

        started = time.monotonic()
        result = self._spawn_and_wait(script_path, workdir)
        result.duration_s = time.monotonic() - started

        result.artifacts = self._discover_artifacts(workdir)
        if result.workdir == "":
            result.workdir = str(workdir)
        return result

    async def execute_async(self, code: str, *, run_id: Optional[str] = None) -> ExecutionResult:
        """Non-blocking wrapper around :meth:`execute` for async rollouts."""
        return await asyncio.get_running_loop().run_in_executor(
            None, lambda: self.execute(code, run_id=run_id)
        )

    def cleanup(self) -> None:
        """Delete managed scratch directories unless retention was requested."""
        if self.keep_workdirs:
            return
        for d in self._managed_dirs:
            # Depth-first (reverse lexicographic) so children vanish before parents.
            for p in sorted(d.rglob("*"), reverse=True):
                with contextlib.suppress(OSError):
                    if p.is_file() or p.is_symlink():
                        p.unlink()
                    elif p.is_dir():
                        p.rmdir()
            with contextlib.suppress(OSError):
                d.rmdir()
        self._managed_dirs.clear()

    def __enter__(self) -> CodeExecutor:
        return self

    def __exit__(self, *_exc_info: Any) -> None:
        self.cleanup()

    # ------------------------------------------------------------------ #
    # Internals                                                          #
    # ------------------------------------------------------------------ #

    def _build_env(self, workdir: Path) -> Dict[str, str]:
        env = dict(_SANDBOX_ENV)
        env["PATH"] = os.environ.get("PATH", os.defpath)
        env["HOME"] = str(workdir)
        env["TMPDIR"] = str(workdir)
        return env

    def _spawn_and_wait(self, script_path: Path, workdir: Path) -> ExecutionResult:
        # Rlimits are applied by the trusted in-child bootstrap (never via
        # preexec_fn, which is unsafe after fork in multithreaded parents);
        # start_new_session detaches the child into its own process group so
        # timeouts can kill the whole tree with one SIGKILL.
        cpu_cap = max(int(self.timeout_s) * 2 + 10, 30)
        mem_bytes = int(self.mem_limit_mb) * 1024 * 1024 if self.mem_limit_mb else 0
        cmd = [
            self.python_executable,
            "-I",
            "-B",
            "-c",
            _SANDBOX_BOOTSTRAP,
            str(mem_bytes),
            str(cpu_cap),
            str(script_path),
        ]
        popen_kwargs: Dict[str, Any] = dict(
            cwd=str(workdir),
            env=self._build_env(workdir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if os.name == "posix":
            popen_kwargs["start_new_session"] = True

        proc = subprocess.Popen(cmd, **popen_kwargs)
        timed_out = False
        try:
            stdout, stderr = proc.communicate(timeout=self.timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            self._kill_process_group(proc)
            stdout, stderr = "", ""
            with contextlib.suppress(Exception):
                stdout, stderr = proc.communicate(timeout=5)

        stderr_text = _truncate(stderr or "")
        error_kind = self._classify(timed_out, proc.returncode, stderr_text)
        summary = _last_traceback_line(stderr_text) if error_kind is not ErrorKind.NONE else None
        return ExecutionResult(
            ok=(not timed_out) and proc.returncode == 0 and error_kind is ErrorKind.NONE,
            error_kind=error_kind,
            returncode=proc.returncode,
            stdout=_truncate(stdout or ""),
            stderr=stderr_text,
            duration_s=0.0,
            timed_out=timed_out,
            workdir=str(workdir),
            error_summary=summary,
        )

    @staticmethod
    def _kill_process_group(proc: subprocess.Popen) -> None:  # pragma: no cover
        """Terminate the whole process tree on timeout."""
        with contextlib.suppress(ProcessLookupError, PermissionError):
            if os.name == "posix":
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            else:
                proc.kill()
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.communicate(timeout=5)

    @staticmethod
    def _classify(timed_out: bool, returncode: Optional[int], stderr: str) -> ErrorKind:
        """Map observable failure signals onto a structured error kind."""
        if timed_out:
            return ErrorKind.TIMEOUT
        markers = (
            (ErrorKind.MEMORY, ("MemoryError", "Cannot allocate memory", "std::bad_alloc")),
            (ErrorKind.IMPORT, ("ModuleNotFoundError", "ImportError")),
            (ErrorKind.SYNTAX, ("SyntaxError",)),
        )
        for kind, needles in markers:
            if any(needle in stderr for needle in needles):
                return kind
        if returncode not in (None, 0):
            return ErrorKind.RUNTIME
        return ErrorKind.NONE

    def _discover_artifacts(self, workdir: Path) -> List[Artifact]:
        """Scan the workdir for exported state files (newest first)."""
        found: List[Tuple[float, Path]] = []
        for pattern in self.artifact_globs:
            for path in workdir.glob(pattern):
                if path.is_file() or path.is_dir():
                    found.append((path.stat().st_mtime, path))
        found.sort(reverse=True)
        seen: set = set()
        artifacts: List[Artifact] = []
        for _, path in found:
            if path in seen:
                continue
            seen.add(path)
            artifacts.append(Artifact(path=str(path), kind=self._artifact_kind(path)))
        return artifacts

    @staticmethod
    def _artifact_kind(path: Path) -> str:
        suffix = path.suffix.lower()
        mapping = {
            ".nc": "netcdf",
            ".nc4": "netcdf",
            ".cdf": "netcdf",
            ".h5": "hdf5",
            ".hdf5": "hdf5",
            ".zarr": "zarr",
            ".npz": "numpy_archive",
            ".json": "json",
            ".csv": "csv",
        }
        return mapping.get(suffix, "unknown")

