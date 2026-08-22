"""Seed benchmark tasks for PhysEval-Agent.

Defines three concrete, self-contained benchmark problems plus the
task-specific verifier for the direct-air-capture kinetics problem:

1. ``grid_24h_curtailment`` -- PyPSA 5-bus transmission system with variable
   wind generation and battery storage; requires an economic dispatch without
   line overloads (verified by :class:`PyPSAGridOracle`).
2. ``tracer_advection_2d`` -- 2D passive tracer transport across a vortex wind
   field conserving total scalar mass (verified by :class:`ClimateGridOracle`).
3. ``direct_air_capture_kinetics`` -- temperature-swing adsorption (TSA) cycle
   simulation checking cyclic steady-state mass balance (verified by the local
   :class:`DACKineticsOracle`).

Each task states the *exact* state-file contract (variable names, units,
coordinates) so generated code can be judged deterministically.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from physeval.agent.loop import RolloutTask
from physeval.oracle.base import BasePhysicsOracle, StateFileError, VerificationResult
from physeval.oracle.climate_invariants import ClimateGridOracle
from physeval.oracle.grid_invariants import PyPSAGridOracle

__all__ = [
    "SEED_TASKS",
    "BenchmarkTask",
    "DACKineticsOracle",
    "all_seed_tasks",
    "get_seed_task",
]



class BenchmarkTask(RolloutTask):
    """Seed task with difficulty metadata and an optional starter skeleton."""

    difficulty: str = "medium"
    starter_code: Optional[str] = None


# --------------------------------------------------------------------------- #
# Task-local oracle: DAC temperature-swing adsorption kinetics                #
# --------------------------------------------------------------------------- #


class DACKineticsOracle(BasePhysicsOracle):
    """Verifier for TSA cyclic steady-state mass balance.

    Expected state contract (NetCDF via xarray):

    - dimension ``time``;
    - ``cycle`` ``(time)`` int -- zero-based cycle index per timestep;
    - ``m_adsorbed`` ``(time)`` kg -- instantaneous bed CO2 loading;
    - ``captured_cumulative`` ``(time)`` kg -- monotone cumulative captured mass;
    - ``T_bed`` ``(time)`` K -- optional but bounds-checked when present.

    Invariants:
        1. Non-negative adsorbate loading.
        2. Per-cycle capture consistency: desorbed bed mass equals the window
           increment of ``captured_cumulative``.
        3. Cyclic steady state: cycle-to-cycle drift of end-of-cycle loading
           and captured amount decays below tolerance.
    """

    name: str = "dac-tsa-kinetics"
    supported_extensions: Tuple[str, ...] = (".nc", ".nc4", ".cdf")

    def __init__(
        self,
        *,
        balance_atol_kg: float = 1e-3,
        balance_rtol: float = 1e-4,
        steady_state_rtol: float = 5e-3,
        min_cycles: int = 3,
    ) -> None:
        """Configure tolerances.

        Args:
            balance_atol_kg: Absolute per-cycle capture residual tolerance.
            balance_rtol: Relative residual tolerance scaled by desorbed mass.
            steady_state_rtol: Allowed fractional cycle-to-cycle drift.
            min_cycles: Minimum completed cycles required to judge steady state.
        """
        self.balance_atol_kg = float(balance_atol_kg)
        self.balance_rtol = float(balance_rtol)
        self.steady_state_rtol = float(steady_state_rtol)
        self.min_cycles = int(min_cycles)

    def verify(self, state_file_path: str) -> VerificationResult:
        """Check loading bounds, per-cycle capture balance, and steady state."""
        self.validate_path(state_file_path)
        try:
            import xarray as xr
        except ImportError as exc:
            raise StateFileError(
                "Optional dependency 'xarray' is required to verify DAC states."
            ) from exc

        try:
            ds = xr.open_dataset(state_file_path)
        except Exception as exc:
            raise StateFileError(f"Cannot open DAC state {state_file_path!r}: {exc}") from exc

        violations: List[Any] = []
        metrics: Dict[str, float] = {}

        def fatal(name: str, observed: float, threshold: float, msg: str) -> None:
            violations.append(self.make_violation(name, "FATAL", observed, threshold, msg))

        missing = [v for v in ("cycle", "m_adsorbed", "captured_cumulative") if v not in ds]
        if missing:
            present = sorted(map(str, ds.data_vars))
            ds.close()
            fatal(
                "state_contract_missing_variables",
                float(len(missing)),
                0.0,
                f"DAC state is missing required variables {missing}; "
                f"dataset contains {present}.",
            )
            return VerificationResult(passed=False, violations=violations, metrics=metrics)

        cycle = np.asarray(ds["cycle"].values).astype(int).ravel()
        m_ads = np.asarray(ds["m_adsorbed"].values, dtype=float).ravel()
        captured = np.asarray(ds["captured_cumulative"].values, dtype=float).ravel()
        if "T_bed" in ds:
            t_bed = np.asarray(ds["T_bed"].values, dtype=float).ravel()
            finite_t = t_bed[np.isfinite(t_bed)]
            if finite_t.size:
                metrics["min_T_bed_K"] = float(finite_t.min())
                metrics["max_T_bed_K"] = float(finite_t.max())
                if finite_t.min() <= 0.0:
                    fatal(
                        "temperature_positive",
                        float(finite_t.min()),
                        0.0,
                        f"Bed temperature must remain above 0 K "
                        f"(observed min={float(finite_t.min()):.6g} K).",
                    )
        ds.close()

        if m_ads.size == 0 or captured.size != m_ads.size or cycle.size != m_ads.size:
            fatal(
                "state_contract_shape_mismatch",
                float(m_ads.size),
                float(captured.size),
                "cycle / m_adsorbed / captured_cumulative must share one non-empty time axis.",
            )
            return VerificationResult(passed=False, violations=violations, metrics=metrics)

        unique_cycles = np.unique(cycle)
        metrics["num_cycles"] = float(unique_cycles.size)
        metrics["final_loading_kg"] = float(m_ads[-1])

        # ---- Invariant 1: physical bounds ---------------------------------
        if float(m_ads.min()) < -1e-12:
            fatal(
                "adsorbate_nonnegative",
                float(m_ads.min()),
                0.0,
                f"Bed loading went negative (min={float(m_ads.min()):.6g} kg); "
                "kinetic integration must be clipped at physical zero.",
            )

        # ---- Invariant 2: per-cycle capture balance ------------------------
        worst_residual = 0.0
        worst_msg = ""
        cap_increments: List[float] = []
        load_deltas: List[float] = []
        desorbed_amounts: List[float] = []
        for c in unique_cycles:
            idx = np.where(cycle == c)[0]
            dm = np.diff(m_ads[idx])
            desorbed = float(np.clip(-dm, 0.0, None).sum())
            load_deltas.append(float(m_ads[idx][-1]) - float(m_ads[idx][0]))
            desorbed_amounts.append(desorbed)
            increment = float(captured[idx[-1]] - captured[idx[0]])
            cap_increments.append(increment)
            residual = abs(increment - desorbed)
            if residual > self.balance_atol_kg + self.balance_rtol * max(desorbed, 1e-9):
                worst_residual = max(worst_residual, residual)
                worst_msg = (
                    f"Cycle {int(c)}: captured_cumulative grew by {increment:.6f} kg "
                    f"but {desorbed:.6f} kg were desorbed from the bed "
                    f"(residual {residual:.6g} kg). Every kilogram released during "
                    "regeneration must appear in captured_cumulative."
                )
        metrics["max_capture_balance_residual_kg"] = worst_residual
        if worst_msg:
            fatal("cyclic_mass_balance", worst_residual, self.balance_atol_kg, worst_msg)

        # ---- Invariant 3: cyclic steady state ------------------------------
        if unique_cycles.size >= max(2, self.min_cycles):
            throughput_scale = max(abs(cap_increments[-1]), desorbed_amounts[-1], 1e-9)
            drift_load = abs(load_deltas[-1]) / throughput_scale
            prev_cap = abs(cap_increments[-2])
            last_cap = abs(cap_increments[-1])
            drift_cap = abs(last_cap - prev_cap) / max(prev_cap, 1e-9)
            metrics["steady_state_drift_fraction"] = max(drift_load, drift_cap)
            if drift_load > self.steady_state_rtol or drift_cap > self.steady_state_rtol:
                fatal(
                    "cyclic_steady_state",
                    max(drift_load, drift_cap),
                    self.steady_state_rtol,
                    f"Cycle-to-cycle drift not converged: loading drift={drift_load:.4g}, "
                    f"capture drift={drift_cap:.4g} exceed rtol={self.steady_state_rtol}. "
                    "Run additional cycles or tighten the kinetic integration step.",
                )
        else:
            violations.append(
                self.make_violation(
                    "insufficient_cycles",
                    "WARNING",
                    float(unique_cycles.size),
                    float(self.min_cycles),
                    f"Only {unique_cycles.size} complete cycle(s) simulated; cyclic "
                    f"steady state cannot be certified (need >= {self.min_cycles}).",
                )
            )

        has_fatal = any(v.severity == "FATAL" for v in violations)
        return VerificationResult(passed=not has_fatal, violations=violations, metrics=metrics)


# --------------------------------------------------------------------------- #
# Starter-code skeletons embedded in prompts (optional scaffolding)           #
# --------------------------------------------------------------------------- #

_GRID_STARTER = '''\
import numpy as np
import pandas as pd
import pypsa

snapshots = pd.date_range("2030-01-01", periods=24, freq="h")
n = pypsa.Network(snapshots=snapshots)

n.add("Bus", [f"bus{i}" for i in range(5)], v_nom=380.0)
# TODO: ring lines bus0-bus1-...-bus4-bus0 with consistent reactance and s_nom limits
# TODO: loads, conventional generators, wind farm with p_max_pu profile, battery storage unit
# TODO: marginal costs so dispatch is economic; wind curtails before expensive units run

n.optimize()          # PyPSA >= 0.29 (use n.lopf() on older releases)
n.export_to_netcdf("network_state.nc")
'''

_TRACER_STARTER = '''\
import numpy as np
import xarray as xr

nx = ny = 64
L = 1.0e6                 # meters
dx = dy = L / nx
dt = 600.0                # seconds
nt = 200

x = (np.arange(nx) + 0.5) * dx
y = (np.arange(ny) + 0.5) * dy
X, Y = np.meshgrid(x, y)

psi = 1.0e5 * np.sin(np.pi * X / L) * np.sin(np.pi * Y / L)   # streamfunction [m^2/s]
u = -(np.gradient(psi, dy, axis=0))     # TODO: verify sign conventions
v = +(np.gradient(psi, dx, axis=1))

tracer = np.exp(-((X - L / 4) ** 2 + (Y - L / 4) ** 2) / (2.0 * (L / 20) ** 2))
# TODO: flux-form advection loop filling `frames` (upwind or MPDATA), periodic
#       boundaries; assert max|u| * dt / dx <= 1 before stepping.
frames = np.zeros((nt + 1, ny, nx))
frames[0] = tracer

ds = xr.Dataset(
    {"tracer": (("time", "y", "x"), frames)},
    coords={"x": ("x", x, {"units": "m"}), "y": ("y", y, {"units": "m"}),
            "time": np.arange(nt + 1) * dt},
)
ds["u"] = (("y", "x"), u)
ds["v"] = (("y", "x"), v)
ds.to_netcdf("simulation_state.nc")
'''

_DAC_STARTER = '''\
import numpy as np
import xarray as xr

q_max = 3.0        # mol/kg equilibrium capacity
k_ref = 2.0e-3     # linear-driving-force rate constant [1/s]

def q_eq(T, p_co2):        # TODO: isotherm, e.g. Toth/Langmuir + van't Hoff T-dependence
    ...

t_cycle = 8 * 3600.0       # 6 h adsorption + 2 h desorption
n_cycles = 6
dt_step = 10.0             # integration step [s]

# TODO: integrate dm/dt = k(T) * (q_eq(T, p_co2) - m) with piecewise T(t);
#       accumulate captured_cumulative from desorbed flux only during regeneration.

ds = xr.Dataset({
    "cycle": (("time",), cycle_index),
    "m_adsorbed": (("time",), m_ads),
    "captured_cumulative": (("time",), captured),
    "T_bed": (("time",), T),
})
ds.to_netcdf("dac_state.nc")
'''


# --------------------------------------------------------------------------- #
# Seed registry                                                               #
# --------------------------------------------------------------------------- #

#: Canonical seed task identifiers (resolved lazily via :func:`get_seed_task`).
SEED_TASKS: Tuple[str, ...] = (
    "grid_24h_curtailment",
    "tracer_advection_2d",
    "direct_air_capture_kinetics",
)


def _build_seed_tasks() -> Tuple[BenchmarkTask, ...]:
    """Instantiate the canonical seed suite (fresh objects each call)."""
    grid_task = BenchmarkTask(
        id="grid_24h_curtailment",
        title="24-hour economic dispatch of a 5-bus wind+battery network",
        difficulty="hard",
        description="""\
Build and solve a 24-hour security-constrained economic dispatch of a
5-bus transmission system in PyPSA and export the solved network.

Topology: five buses `bus0..bus4`, v_nom=380 kV, connected as a ring
(bus0-bus1, bus1-bus2, bus2-bus3, bus3-bus4, bus4-bus0). Give every line a
consistent reactance (any physically plausible ohmic value works, r small)
and thermal limit s_nom chosen so no line exceeds 100% loading in the
optimal solution (e.g. 150 MVA on corridors adjacent to the wind bus).

Generation:
- `gas_base` at bus0: p_nom=200 MW, marginal_cost=55 EUR/MWh.
- `coal_mid` at bus2: p_nom=180 MW, marginal_cost=30 EUR/MWh.
- `wind_farm` at bus4: p_nom=250 MW with an hourly p_max_pu capacity-factor
  profile over 24 h peaking at night (values up to ~0.85), i.e. cheap energy
  that must be curtailed rather than stored beyond feasibility.
Storage:
- `battery` at bus2: PyPSA StorageUnit, p_nom=60 MW, max_hours=4,
  efficiency_store=efficiency_dispatch=0.95, standing_loss=0,
  state_of_charge_initial=40 MWh, cyclic operation (SOC returns to its
  initial value within solver tolerance).
Load: constant hourly demand [80, 120, 90, 70, 60] MW at buses 0..4.

Solve with `n.optimize()` (PyPSA>=0.29; older releases use
`n.lopf(pyomo=False)`) under hourly snapshot weightings, then export with
`n.export_to_netcdf("network_state.nc")`.
""",
        requirements=[
            "Nodal active-power balance holds at every bus/snapshot within 1e-4 MW.",
            "No line or transformer exceeds 100% of its thermal limit.",
            "0 <= P_gen <= p_nom * p_max_pu(t) for every generator and snapshot.",
            "Battery SOC recursion holds across snapshots including store/dispatch efficiencies.",
            "Wind curtails when cheaper than storage round-trip losses plus congestion.",
        ],
        artifact_filename="network_state.nc",
        tags=["pypsa", "power-systems", "curtailment"],
        starter_code=_GRID_STARTER,
        oracle=PyPSAGridOracle(),
    )

    tracer_task = BenchmarkTask(
        id="tracer_advection_2d",
        title="Conservative 2D passive-tracer transport in a vortex wind field",
        difficulty="medium",
        description="""\
Simulate 2D passive-tracer advection by a steady vortex wind field on a
periodic square domain and export the trajectory as NetCDF.

Domain: square side L=1e6 m discretized on a 64x64 cell-centered grid
(dx=dy=L/64). Wind: streamfunction psi(x,y)=1e5*sin(pi*x/L)*sin(pi*y/L)
m^2/s with u=-d(psi)/dy and v=+d(psi)/dx (centered differences suffice).
Time: dt=600 s for nt=200 steps (save all nt+1 frames).
Initial condition: Gaussian blob centered at (L/4, L/4) with sigma=L/20.

Numerics: use a FLUX-FORM conservative scheme (first-order upwind is
acceptable; MPDATA/MacCormack preferred) with periodic boundary conditions.
Before integrating, compute C = max|u|*dt/dx and only proceed if C<=1 --
fail loudly rather than silently degrading.

Export contract (exact names):
- data variable `tracer` with dims (time, y, x), non-negative everywhere;
- wind components `u`, `v` with dims (y, x);
- coordinates `x`, `y` in meters (attrs {'units': 'm'}) and numeric `time`
  in seconds spanning 0..nt*dt;
written via `ds.to_netcdf("simulation_state.nc")`.

The verifier integrates total tracer mass at the first and last frames;
relative change must stay below 1e-5.
""",
        requirements=[
            "Global tracer mass conserved to |dM/M| <= 1e-5 between first and last frame.",
            "Tracer stays non-negative throughout (no undershoot below zero).",
            "CFL number C = u_max*dt/dx <= 1.0 for the chosen step.",
            "Periodic boundaries handled without mass leakage.",
            "Exact variable/coordinate names per the export contract.",
        ],
        artifact_filename="simulation_state.nc",
        tags=["xarray", "climate", "advection"],
        starter_code=_TRACER_STARTER,
        oracle=ClimateGridOracle(
            dt_seconds=600.0,
            tracer_candidates=("tracer",),
            u_wind_candidates=("u",),
            v_wind_candidates=("v",),
        ),
    )

    dac_task = BenchmarkTask(
        id="direct_air_capture_kinetics",
        title="Temperature-swing adsorption (TSA) DAC cycle to cyclic steady state",
        difficulty="hard",
        description="""\
Simulate a solid-sorbent direct-air-capture bed operating repeated
temperature-swing adsorption cycles until cyclic steady state, exporting
the full time series.

Cycle recipe (repeat n_cycles=6 times):
- Adsorption stage: 6 h at T_ads=298 K with feed p_co2=4e-4 bar.
- Desorption stage: 2 h at T_des=393 K under purge/vacuum; ALL CO2 released
  during this stage counts as captured.

Kinetics: linear driving force dm/dt = k(T)*(q_eq(T,p_co2) - m), with a
Toth or Langmuir isotherm q_eq(T,p) whose affinity decreases with
temperature (van't Hoff effect) and k(T) ~ 2e-3 1/s at 298 K with mild
Arrhenius increase. Integrate with dt <= 10 s (explicit Euler acceptable
given the small step).

Bookkeeping (all on a common `time` dimension):
- `cycle`: integer running cycle index per timestep;
- `m_adsorbed`: instantaneous bed loading [kg], never negative;
- `captured_cumulative`: monotone non-decreasing cumulative captured mass
  [kg]; increments ONLY during desorption stages;
- `T_bed`: bed temperature [K].

Export with `ds.to_netcdf("dac_state.nc")`. The verifier checks per-cycle
capture balance (desorbed == captured increment) and cycle-to-cycle drift
(cyclic steady state).
""",
        requirements=[
            "Per-cycle mass balance: desorbed bed mass equals captured_cumulative increment within tolerance.",
            "Bed loading remains physically bounded (>= 0) at every timestep.",
            "Cyclic steady state reached: cycle-to-cycle drift <= 0.5% once enough cycles are run.",
            "Temperature stays positive and near the stated stage temperatures.",
            "At least 3 complete cycles recorded.",
        ],
        artifact_filename="dac_state.nc",
        tags=["kinetics", "dac", "tsa"],
        starter_code=_DAC_STARTER,
        oracle=DACKineticsOracle(),
    )

    return (grid_task, tracer_task, dac_task)


def get_seed_task(task_id: str) -> BenchmarkTask:
    """Return a fresh instance of seed task *task_id*.

    Raises:
        KeyError: If the id is unknown (valid ids listed in the message).
    """
    tasks = {t.id: t for t in _build_seed_tasks()}
    if task_id not in tasks:
        raise KeyError(f"Unknown task id {task_id!r}; valid ids: {sorted(tasks)}")
    return tasks[task_id]


def all_seed_tasks(ids: Optional[List[str]] = None) -> List[BenchmarkTask]:
    """Return seed tasks, optionally filtered in the order given by *ids*."""
    if ids is None:
        return list(_build_seed_tasks())
    return [get_seed_task(i) for i in ids]


if __name__ == "__main__":  # pragma: no cover - manual inspection helper
    for _t in _build_seed_tasks():
        print(f"{_t.id}: {_t.title} [{_t.difficulty}] -> {_t.artifact_filename}")
