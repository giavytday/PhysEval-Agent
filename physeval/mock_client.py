"""Deterministic offline stand-in for an OpenAI-compatible chat client.

:class:`MockChatClient` lets the whole PhysEval pipeline (rollouts, benchmark
evaluation) run hermetically -- no network, no API key -- while still
exercising the complete Generate -> Execute -> Verify -> Correct loop:

* the **first attempt** per domain is intentionally *physically flawed*
  (scaled tracer frame breaking mass conservation, flat capture bookkeeping
  breaking cyclic balance, undersized line limits breaking thermal bounds), so
  oracles produce real FATAL violations;
* any **repair turn** (detected via the ``## Diagnostic report`` marker the
  agent loop injects) receives the corrected solution, which passes its
  domain oracle.

Grid solutions require PyPSA inside the sandbox; when it is absent the script
exits with a clear message, which keeps smoke runs meaningful on minimal
environments while still producing structured execution diagnostics.

This module lives in the library (rather than tests/) because the shell
pipeline runner and CI both rely on ``--client mock`` for reproducible,
network-free integration runs.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

LOGGER = logging.getLogger("physeval.mock_client")

_REPAIR_MARKER = "## Diagnostic report"


# --------------------------------------------------------------------------- #
# Parameter-aware solution builders                                           #
#                                                                             #
# The mock must generate states that are consistent with the *specific*       #
# task statement (grid size, timestep, cycle count), so it parses the         #
# constants out of the generation prompt instead of hard-coding them.         #
# --------------------------------------------------------------------------- #

def _climate_params(prompt: str) -> Tuple[float, int, int, bool]:
    """Extract (dt_seconds, nx, ny, is_3d) from a generation prompt."""
    def _match(pattern: str, default: float) -> float:
        found = re.search(pattern, prompt)
        return float(found.group(1)) if found else default

    grid_match = re.search(r"grid (\d+) x (\d+)", prompt)
    nx = int(grid_match.group(1)) if grid_match else 32
    ny = int(grid_match.group(2)) if grid_match else 32
    dt = _match(r"dt = ([0-9][0-9.]*) s", 600.0)
    side_m = _match(r"L=([0-9][0-9.]*) m", 1_000_000.0)
    is_3d = "vertical levels" in prompt
    return dt, max(nx, 8), max(ny, 8), side_m if side_m > 0 else 1.0e6, is_3d


def _build_climate_code(*, broken: bool, dt: float, nx: int, ny: int,
                        side_m: float, is_3d: bool) -> str:
    """Self-consistent tracer solution for the stated grid/timestep.

    A uniform tracer field under a flux-form update with periodic boundaries
    conserves mass exactly; the wind magnitude is derived from the task's own
    ``dt`` and ``dx`` such that the Courant number equals 0.4.
    """
    dx = side_m / nx
    u_max = max(min(5.0, 0.4 * dx / dt), 1e-9)
    bug_line = (
        "\nframes[-1] *= 1.02   # BUG: non-conservative rescaling of final frame"
        if broken else ""
    )
    lead = "8, " if is_3d else ""
    wind_shape = "(8, ny, nx)" if is_3d else "(ny, nx)"
    tracer_dims = '"time", "z", "y", "x"' if is_3d else '"time", "y", "x"'
    wind_dims = '"z", "y", "x"' if is_3d else '"y", "x"'
    z_lines = (
        'ds["z"] = ("z", np.arange(8) * 100.0)\n'
        'ds["z"].attrs["units"] = "m"\n'
    ) if is_3d else ""

    return f'''\
"""Reference conservative tracer transport solution."""
import numpy as np
import xarray as xr

nx, ny = {nx}, {ny}
L = {side_m!r}
dx = dy = L / nx
dt = {dt!r}
nt = 4

x = (np.arange(nx) + 0.5) * dx
y = (np.arange(ny) + 0.5) * dy

# Uniform field + flux-form + periodic BCs => exactly mass-conserving.
frames = np.full((nt + 1, {lead}ny, nx), 0.7){bug_line}

u_max = {u_max!r}   # C = u_max*dt/dx <= 0.4 by construction
u = np.full({wind_shape}, u_max)
v = np.zeros_like(u)

ds = xr.Dataset()
ds["tracer"] = (({tracer_dims}), frames)
ds["u"] = (({wind_dims}), u)
ds["v"] = (({wind_dims}), v)
ds["x"] = ("x", x)
ds["y"] = ("y", y)
ds["x"].attrs["units"] = "m"
ds["y"].attrs["units"] = "m"
{z_lines}ds["time"] = ("time", np.arange(nt + 1) * dt)

print("frames:", frames.shape)
ds.to_netcdf("simulation_state.nc")
'''


def _build_kinetics_code(*, broken: bool, n_cycles: int) -> str:
    """TSA cycle bookkeeping that satisfies the state contract exactly.

    ``captured_cumulative`` increments mirror the desorbed bed mass step by
    step, so the per-cycle balance always holds; when ``broken`` the loading
    baseline creeps upward every cycle, tripping the cyclic steady-state
    invariant while leaving everything else physically valid.
    """
    n_cycles = max(3, int(n_cycles))
    cycle_idx: List[int] = []
    loading: List[float] = []
    captured: List[float] = []
    t_bed: List[float] = []
    total = 0.0
    ads_shape = [1.0, 1.2, 1.4, 1.6]
    des_tail = [1.4, 1.2]
    for c in range(n_cycles):
        d = (0.08 * c) if broken else 0.0
        seq = [v + d for v in ads_shape] + [1.6 + d] + [v + d for v in des_tail] \
            + [1.0 + 2.0 * d]
        baseline = total
        n_ads = len(ads_shape)
        for i, value in enumerate(seq):
            cycle_idx.append(c)
            loading.append(round(value, 6))
            t_bed.append(298.0 if i < n_ads else 393.0)
            if i < n_ads:
                captured.append(round(baseline, 6))
            else:
                drop = seq[i - 1] - value
                total += max(drop, 0.0)
                captured.append(round(total, 6))

    return f'''\
"""TSA cyclic steady-state reference solution."""
import numpy as np
import xarray as xr

cycle_index = {cycle_idx!r}
loading = {loading!r}
captured_cumulative = {captured!r}
bed_temperature = {t_bed!r}

assert min(loading) >= 0.0, "loading must stay non-negative"
assert all(b >= a for a, b in zip(captured_cumulative, captured_cumulative[1:])), \\
    "captured_cumulative must be monotone"

ds = xr.Dataset()
ds["cycle"] = (("time",), np.array(cycle_index, dtype=np.int64))
ds["m_adsorbed"] = (("time",), np.array(loading))
ds["captured_cumulative"] = (("time",), np.array(captured_cumulative))
ds["T_bed"] = (("time",), np.array(bed_temperature))
print("cycles:", cycle_index[-1] + 1)
ds.to_netcdf("dac_state.nc")
'''

_GRID_PREAMBLE = '''\
"""PyPSA grid dispatch solution (requires pypsa in the sandbox)."""
try:
    import pandas as pd
    import pypsa
except ImportError as exc:
    raise SystemExit(
        "grid tasks need PyPSA installed in the sandbox environment; "
        "pip install 'physeval-agent[grid]'"
    ) from exc

snapshots = pd.date_range("2030-01-01", periods=2, freq="h")
n = pypsa.Network(snapshots=snapshots)
n.add("Bus", ["busA", "busB"], v_nom=380.0)
'''

_GRID_GOOD = (
    _GRID_PREAMBLE
    + '''\
n.add("Generator", "gas", bus="busA", p_nom=120.0, marginal_cost=55.0)
n.add("Load", "city", bus="busB", p_set=[80.0, 80.0])
n.add("Line", "tie", bus0="busA", bus1="busB", x=0.05, r=0.01,
      s_nom=150.0, s_max_pu=1.0)
n.generators_t.p_set.loc[:, "gas"] = [100.0, 100.0]
n.add("StorageUnit", "battery", bus="busA", p_nom=60.0, max_hours=4,
      efficiency_store=0.95, efficiency_dispatch=0.95, standing_loss=0.0,
      state_of_charge_initial=40.0)
n.storage_units_t.p_set.loc[:, "battery"] = [-20.0, -20.0]

# A linear power flow derives physically consistent branch flows and battery
# SOC from the fixed dispatch, so every oracle invariant holds by construction.
n.lpf()
print("dispatch consistent; exporting solved state")
n.export_to_netcdf("network_state.nc")
'''
)

_GRID_BROKEN = (
    _GRID_PREAMBLE
    + '''\
n.add("Generator", "gas", bus="busA", p_nom=120.0, marginal_cost=55.0)
n.add("Load", "city", bus="busB", p_set=[80.0, 80.0])
n.add("Line", "tie", bus0="busA", bus1="busB", x=0.05, r=0.01,
      s_nom=1.0, s_max_pu=1.0)   # BUG: absurdly tight thermal limit
n.generators_t.p_set.loc[:, "gas"] = [100.0, 100.0]
n.add("StorageUnit", "battery", bus="busA", p_nom=60.0, max_hours=4,
      efficiency_store=0.95, efficiency_dispatch=0.95, standing_loss=0.0,
      state_of_charge_initial=40.0)
n.storage_units_t.p_set.loc[:, "battery"] = [-20.0, -20.0]
n.lpf()
n.export_to_netcdf("network_state.nc")
'''
)


def _domain_of(messages: List[Dict[str, str]]) -> Optional[str]:
    """Extract the physics domain from the generation prompt's task id."""
    for message in messages:
        content = message.get("content") or ""
        if "Task id:" not in content:
            continue
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith("Task id:"):
                task_id = stripped.split("Task id:", 1)[1].strip()
                if task_id.startswith("grid"):
                    return "grid"
                if task_id.startswith("climate"):
                    return "climate"
                if task_id.startswith("kinetics"):
                    return "kinetics"
                return None
    return None


def _is_repair_turn(messages: List[Dict[str, str]]) -> bool:
    """True when the latest user message carries oracle diagnostics."""
    last_user = next(
        (m.get("content") or "" for m in reversed(messages) if m.get("role") == "user"),
        "",
    )
    return _REPAIR_MARKER in last_user


def _render(content: str) -> str:
    return f"Here is my implementation:\n```python\n{content}\n```\nDone."


class MockChatClient:
    """Drop-in async stand-in exposing ``chat.completions.create``."""

    def __init__(self) -> None:
        self.request_count = 0
        outer = self

        class _Completions:
            async def create(self, *, model: str = "mock-model",
                             messages: Optional[List[Dict[str, str]]] = None,
                             temperature: float = 0.0, **_kwargs: Any) -> Any:
                return outer._create(model=model, messages=messages,
                                     temperature=temperature, **_kwargs)

        class _Chat:
            def __init__(self) -> None:
                self.completions = _Completions()

        self.chat = _Chat()

    def _create(self, *, model: str = "mock-model",
                messages: Optional[List[Dict[str, str]]] = None,
                temperature: float = 0.0, **_kwargs: Any) -> Any:
        assert messages is not None
        self.request_count += 1
        domain = _domain_of(messages)
        repair = _is_repair_turn(messages)
        prompt_text = "\n".join(str(m.get("content") or "") for m in messages)

        if domain == "grid":
            code = _GRID_GOOD if repair else _GRID_BROKEN
        elif domain == "kinetics":
            cycles_match = re.search(r"n_cycles=(\d+)", prompt_text)
            n_cycles = int(cycles_match.group(1)) if cycles_match else 6
            code = _build_kinetics_code(broken=not repair, n_cycles=n_cycles)
        else:  # climate, plus unknown domains (which get the passing variant)
            dt, nx, ny, side_m, is_3d = _climate_params(prompt_text)
            code = _build_climate_code(
                broken=(domain == "climate" and not repair),
                dt=dt, nx=nx, ny=ny, side_m=side_m, is_3d=is_3d,
            )
        LOGGER.debug("mock call #%d domain=%s repair=%s", self.request_count, domain, repair)
        return _MockResponse(_render(code))


class _MockMessage:
    def __init__(self, content: str) -> None:
        self.content = content
        self.role = "assistant"


class _MockChoice:
    def __init__(self, content: str) -> None:
        self.message = _MockMessage(content)
        self.finish_reason = "stop"


class _MockUsage:
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0


class _MockResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_MockChoice(content)]
        self.usage = _MockUsage()
        self.model = "mock-model"
        self.id = "mock-completion"
