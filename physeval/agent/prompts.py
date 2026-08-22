"""Prompt templates and structured-output instructions for PhysEval-Agent.

The agent speaks two prompt dialects:

* *generation*: full task specification plus an output contract demanding one
  deterministic, self-contained Python script that exports a serialized state
  artifact;
* *repair*: a structured JSON diagnostic payload (exact violation metrics,
  power imbalances, Courant numbers, stderr excerpts) plus instructions to
  reply with strict JSON containing the patched script.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence

__all__ = [
    "JSON_REPAIR_INSTRUCTIONS",
    "OUTPUT_CONTRACT",
    "SYSTEM_PROMPT",
    "build_generation_prompt",
    "build_repair_prompt",
    "extract_code",
]

SYSTEM_PROMPT: str = """\
You are PhysEval-Agent, an expert scientific-computing engineer specializing in \
power-systems modeling (PyPSA), climate/ocean simulation (NumPy/xarray), and \
numerical methods.

You write production-grade Python that satisfies physical conservation laws
exactly. You favor flux-form discretizations, implicit solvers, and defensive
numerical hygiene over quick approximations. Your code is always deterministic:
fixed seeds, single-threaded BLAS, no network access, no wall-clock dependence.
"""

OUTPUT_CONTRACT: str = """\
## Output contract (mandatory)

1. Reply with exactly ONE fenced Python code block (```python ... ```) and
   nothing else before or after it.
2. The block contains a COMPLETE, self-contained script -- never fragments,
   pseudocode, diffs, or REPL transcripts.
3. The script writes its serialized simulation state to the exact relative
   path `{artifact}` inside the current working directory before exiting.
4. It must terminate well within the execution timeout and print at most a few
   short progress lines to stdout.
5. No interactive input, no network calls, no filesystem writes outside the
   working directory.
"""

_GENERATION_TEMPLATE: str = """\
# Benchmark task

Task id: {task_id}
Title: {title}

{description}

## Requirements
{requirements}

{contract}
Artifact to produce: `{artifact}`
"""


def format_requirements(requirements: Sequence[str]) -> str:
    """Render requirement strings as a numbered markdown list."""
    return "\n".join(f"{i}. {r}" for i, r in enumerate(requirements, start=1))


def build_generation_prompt(
    task_id: str,
    title: str,
    description: str,
    requirements: Sequence[str],
    artifact_filename: str,
) -> str:
    """Compose the initial code-generation prompt for a benchmark task."""
    return _GENERATION_TEMPLATE.format(
        task_id=task_id,
        title=title,
        description=description.strip(),
        requirements=format_requirements(requirements),
        contract=OUTPUT_CONTRACT.format(artifact=artifact_filename),
        artifact=artifact_filename,
    )


_JSON_PAYLOAD_HEADER: str = """\
## Diagnostic report (turn {turn})

The previous submission did **not** satisfy the task. Structured findings:

```json
{payload}
```
"""

_REPAIR_TEMPLATE: str = """\
{_json_header}
### Execution transcript (tail)

stdout:
```
{stdout_tail}
```

stderr:
```
{stderr_tail}
```

{json_instructions}
"""


def build_repair_prompt(
    *,
    turn: int,
    execution: Optional[Mapping[str, Any]],
    verification: Optional[Mapping[str, Any]],
    stdout_tail_chars: int = 4_000,
    stderr_tail_chars: int = 6_000,
) -> str:
    """Build the targeted-patch prompt from sandbox + oracle evidence.

    Args:
        turn: Current rollout turn (1-based, generation included).
        execution: :meth:`ExecutionResult.to_dict` payload, may be ``None``.
        verification: :class:`VerificationResult` dump, may be ``None``.
        stdout_tail_chars / stderr_tail_chars: Transcript tail budgets.

    Returns:
        A markdown prompt embedding the structured JSON error payload.
    """
    payload: Dict[str, Any] = {
        "turn": turn,
        "execution": _slim_execution(execution),
        "oracle_verification": _slim_verification(verification),
    }
    header = _JSON_PAYLOAD_HEADER.format(turn=turn, payload=json.dumps(payload, indent=2))

    stdout_tail = ""
    stderr_tail = ""
    if execution:
        stdout_tail = str(execution.get("stdout") or "")[-stdout_tail_chars:]
        stderr_tail = str(execution.get("stderr") or "")[-stderr_tail_chars:]

    return _REPAIR_TEMPLATE.format(
        _json_header=header,
        stdout_tail=stdout_tail or "(empty)",
        stderr_tail=stderr_tail or "(empty)",
        json_instructions=JSON_REPAIR_INSTRUCTIONS,
    )


JSON_REPAIR_INSTRUCTIONS: str = """\
## Repair instructions (strict)

Reply with ONLY a JSON object -- no prose, no markdown fences around it --
matching exactly this schema:

{
  "diagnosis": "one paragraph root-cause analysis citing the exact metrics above",
  "patch_strategy": "ordered list of concrete numerical/code changes",
  "code": "the COMPLETE corrected Python script as a single string"
}

Rules:
- `code` must be the full runnable script (not a diff) and must still honor
  the original output contract, including the required artifact filename.
- Address every FATAL violation; do not weaken thresholds or tolerances.
- If the failure was a timeout, reduce computational cost rather than skipping
  required physics.
"""


_CODE_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)


def extract_code(response_text: str) -> Optional[str]:
    """Extract the candidate script from an LLM response.

    Strategy order:
      1. Strict-JSON reply with a top-level ``code`` string.
      2. First fenced ```python block.
      3. Bare-heuristic: the entire text if it looks like pure source.

    Returns ``None`` when no plausible program can be recovered.
    """
    text = response_text.strip()
    if text.startswith("{"):
        try:
            obj = json.loads(text)
            code = obj.get("code")
            if isinstance(code, str) and code.strip():
                return code.strip()
        except json.JSONDecodeError:
            pass
    match = _CODE_FENCE_RE.search(response_text)
    if match:
        return match.group(1).strip()
    looks_like_source = text.startswith(("import ", "from ", '"""', "#!", "# "))
    if looks_like_source and "\n" in text:
        return text
    return None


def _slim_execution(execution: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Compact execution payload safe for prompt injection."""
    if not execution:
        return {"status": "no-execution-record"}
    slim_keys = (
        "ok",
        "error_kind",
        "returncode",
        "timed_out",
        "duration_s",
        "error_summary",
        "artifacts",
    )
    out: Dict[str, Any] = {k: execution.get(k) for k in slim_keys}
    out["state_files"] = [
        a.get("path") for a in (execution.get("artifacts") or []) if isinstance(a, Mapping)
    ]
    return out


def _slim_verification(verification: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Compact verification payload preserving exact numeric evidence."""
    if not verification:
        return {"status": "not-run"}
    violations = verification.get("violations") or []
    return {
        "passed": bool(verification.get("passed")),
        "violations": [
            {
                "name": v.get("name"),
                "severity": v.get("severity"),
                "observed_value": v.get("observed_value"),
                "threshold": v.get("threshold"),
                "message": v.get("message"),
            }
            for v in violations
        ],
        "metrics": verification.get("metrics") or {},
    }


def summarize_step_history(steps: List[Mapping[str, Any]], limit: int = 3) -> str:
    """One-line-per-step recap used to anchor multi-turn context."""
    lines: List[str] = []
    for s in steps[-limit:]:
        verdict = "ok" if s.get("exec_ok") else f"exec-error({s.get('error_kind')})"
        lines.append(f"- turn {s.get('turn')}: {verdict}")
    return "\n".join(lines) if lines else "- no prior steps"
