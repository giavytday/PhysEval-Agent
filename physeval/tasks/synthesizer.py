"""Automated physics-task synthesis across three domains.

:class:`PhysicsTaskSynthesizer` programmatically builds hundreds of
parameterized, self-contained benchmark scenarios:

* **Power grids (PyPSA)** -- randomized connected topologies (5-57 buses),
  diurnal wind/solar capacity-factor profiles defined by closed-form formulas,
  battery storage with efficiency and ramp-limit parameters, and explicit line
  thermal limits.
* **Climate/ocean dynamics (xarray)** -- 2D/3D passive-tracer
  advection-diffusion on variable grid resolutions driven by analytic vortex
  streamfunctions, with timestep/grid pairs chosen so the CFL condition is
  satisfiable and global mass conservation is strict.
* **Carbon kinetics** -- multistage temperature/pressure-swing adsorption
  cycles with Arrhenius rate constants and Langmuir equilibrium constraints.

Every scenario is emitted as a :class:`SynthTaskSpec`: a fully serializable
record whose ``description`` embeds every numeric constant (formula-based, so
prompts stay compact even for 57-bus systems) and whose ``oracle_kwargs``
permit deterministic reconstruction of the correct verifier via
:meth:`PhysicsTaskSynthesizer.spec_to_task`. Suites are exported to
``benchmark_suite.jsonl`` and can be reloaded bit-for-bit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
from pydantic import BaseModel, Field

from physeval.oracle.base import BasePhysicsOracle
from physeval.oracle.climate_invariants import ClimateGridOracle
from physeval.oracle.grid_invariants import PyPSAGridOracle
from physeval.tasks.seed_tasks import BenchmarkTask, DACKineticsOracle

__all__ = [
    "DEFAULT_SUITE_PATH",
    "DOMAINS",
    "PhysicsTaskSynthesizer",
    "SynthTaskSpec",
]

#: Canonical domain identifiers.
DOMAINS: Tuple[str, ...] = ("grid", "climate", "kinetics")

#: Default export location: ``physeval/tasks/benchmark_suite.jsonl``.
DEFAULT_SUITE_PATH = Path(__file__).resolve().parent / "benchmark_suite.jsonl"

#: Universal gas constant [J/(mol K)] used in kinetics statements.
_R_GAS = 8.314


def _r(value: float, nd: int = 4) -> float:
    """Round to *nd* decimals and normalize -0.0 for stable serialization."""
    out = round(float(value), nd)
    return out + 0.0


class SynthTaskSpec(BaseModel):
    """Serializable synthesis record (one JSONL line of the suite)."""

    id: str
    domain: Literal["grid", "climate", "kinetics"]
    title: str
    description: str
    requirements: List[str] = Field(default_factory=list)
    artifact_filename: str
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    tags: List[str] = Field(default_factory=list)
    #: Flat parameter snapshot (prefixed keys) for stratified analysis.
    params: Dict[str, float] = Field(default_factory=dict)
    #: Kwargs forwarded to the domain oracle constructor on reload.
    oracle_kwargs: Dict[str, Any] = Field(default_factory=dict)


def _build_oracle(domain: str, oracle_kwargs: Dict[str, Any]) -> BasePhysicsOracle:
    """Instantiate the domain verifier from its serialized kwargs."""
    kwargs = dict(oracle_kwargs)
    if domain == "grid":
        return PyPSAGridOracle(**kwargs)
    if domain == "climate":
        # Narrow variable discovery so heterogeneous suites stay unambiguous.
        kwargs.setdefault("tracer_candidates", ["tracer"])
        kwargs.setdefault("u_wind_candidates", ["u"])
        kwargs.setdefault("v_wind_candidates", ["v"])
        return ClimateGridOracle(**kwargs)
    if domain == "kinetics":
        return DACKineticsOracle(**kwargs)
    raise ValueError(f"Unknown domain {domain!r}")


class PhysicsTaskSynthesizer:
    """Seeded generator of parameterized physics benchmark tasks.

    Example:
        >>> synth = PhysicsTaskSynthesizer(seed=42)
        >>> suite = synth.synthesize(total=600)
        >>> len(suite)
        600
    """

    def __init__(self, *, seed: int = 0) -> None:
        """Create a synthesizer with an isolated deterministic RNG."""
        self._rng = np.random.default_rng(seed)
        self._seed = int(seed)

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #

    def synthesize(self, total: int = 600, *, per_domain: Optional[int] = None) -> List[SynthTaskSpec]:
        """Generate *total* tasks spread evenly across the three domains.

        Args:
            total: Overall task count (>= 3).
            per_domain: Explicit count per domain; overrides even split and
                ignores ``total`` when provided.
        """
        if per_domain is not None:
            if per_domain <= 0:
                raise ValueError("per_domain must be positive.")
            counts = {d: per_domain for d in DOMAINS}
        else:
            if total < len(DOMAINS):
                raise ValueError(f"total must be >= {len(DOMAINS)}.")
            base, rem = divmod(total, len(DOMAINS))
            counts = {d: base + (1 if i < rem else 0) for i, d in enumerate(DOMAINS)}

        specs: List[SynthTaskSpec] = []
        counters: Dict[str, int] = {d: 0 for d in DOMAINS}
        for domain in DOMAINS:
            for _ in range(counts[domain]):
                builder = {
                    "grid": self._build_grid_spec,
                    "climate": self._build_climate_spec,
                    "kinetics": self._build_kinetics_spec,
                }[domain]
                specs.append(builder(counters[domain]))
                counters[domain] += 1
        return specs

    def spec_to_task(self, spec: SynthTaskSpec) -> BenchmarkTask:
        """Materialize a live :class:`BenchmarkTask` (with oracle) from a spec."""
        return BenchmarkTask(
            id=spec.id,
            title=spec.title,
            description=spec.description,
            requirements=spec.requirements,
            artifact_filename=spec.artifact_filename,
            tags=spec.tags,
            difficulty=spec.difficulty,
            starter_code=None,
            oracle=_build_oracle(spec.domain, spec.oracle_kwargs),
        )

    @staticmethod
    def export_jsonl(
        specs: Sequence[SynthTaskSpec],
        path: str | Path = DEFAULT_SUITE_PATH,
    ) -> Path:
        """Write *specs* as JSONL; parent directories are created as needed."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as fh:
            for spec in specs:
                fh.write(json.dumps(spec.model_dump(), ensure_ascii=False, allow_nan=False))
                fh.write("\n")
        return out_path

    @staticmethod
    def load_suite(path: str | Path) -> List[SynthTaskSpec]:
        """Reload a previously exported suite (validating every line)."""
        suite_path = Path(path)
        if not suite_path.is_file():
            raise FileNotFoundError(f"Benchmark suite not found: {suite_path}")
        specs: List[SynthTaskSpec] = []
        with suite_path.open("r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    specs.append(SynthTaskSpec.model_validate(json.loads(line)))
                except Exception as exc:
                    raise ValueError(f"Invalid suite record at line {lineno}: {exc}") from exc
        return specs

    # ------------------------------------------------------------------ #
    # Domain A: power grids                                              #
    # ------------------------------------------------------------------ #

    def _build_grid_spec(self, index: int) -> SynthTaskSpec:
        rng = self._rng
        n_buses = int(rng.integers(5, 58))

        # Ring backbone plus ~20% random chords guarantees connectivity.
        edges = [(i, (i + 1) % n_buses) for i in range(n_buses)]
        n_chords = int(rng.integers(0, max(2, n_buses // 4)))
        seen = set(edges)
        attempts = 0
        while len([e for e in edges]) < n_buses + n_chords and attempts < 200:
            a, b = sorted(int(x) for x in rng.integers(0, n_buses, size=2))
            attempts += 1
            if a != b and (a, b) not in seen:
                edges.append((a, b))
                seen.add((a, b))

        s_nom_base = _r(rng.uniform(80.0, 220.0), 1)
        s_nom_jitter = _r(rng.uniform(0.05, 0.15), 3)
        x_line = _r(rng.uniform(0.03, 0.11), 4)

        gen_bus = int(rng.integers(0, n_buses))
        coal_bus = int(rng.integers(0, n_buses)) if n_buses >= 3 else gen_bus
        gas_pnom = _r(rng.uniform(140.0, 260.0), 1)
        gas_mc = _r(rng.uniform(45.0, 70.0), 1)
        coal_pnom = _r(rng.uniform(120.0, 240.0), 1)
        coal_mc = _r(rng.uniform(22.0, 38.0), 1)
        gas_ramp = _r(rng.uniform(0.25, 0.6), 3)
        coal_ramp = _r(rng.uniform(0.1, 0.35), 3)

        wind_bus = int(rng.integers(0, n_buses))
        wind_pnom = _r(rng.uniform(150.0, 320.0), 1)
        wind_base = _r(rng.uniform(0.18, 0.38), 3)
        wind_amp = _r(rng.uniform(0.28, 0.52), 3)
        wind_peak_h = int(rng.integers(1, 7))          # night peak
        wind_ripple = _r(rng.uniform(0.02, 0.08), 3)

        solar_bus = int(rng.integers(0, n_buses))
        solar_pnom = _r(rng.uniform(100.0, 240.0), 1)
        solar_amp = _r(rng.uniform(0.55, 0.85), 3)

        bat_bus = int(rng.integers(0, n_buses))
        bat_pnom = _r(rng.uniform(30.0, 80.0), 1)
        bat_hours = _r(rng.uniform(3.0, 6.0), 2)
        bat_eff = _r(rng.uniform(0.90, 0.97), 3)
        bat_ramp = _r(rng.uniform(0.3, 0.8), 3)

        load_lo = int(rng.integers(30, 61))
        load_hi = load_lo + int(rng.integers(30, 71))
        load_prime = int(rng.integers(3, 17))
        load_shape_amp = _r(rng.uniform(0.12, 0.3), 3)

        difficulty = "easy" if n_buses <= 9 else ("medium" if n_buses <= 24 else "hard")
        edge_str = " ".join(f"({a},{b})" for a, b in edges)

        description = f"""\
Build and solve a 24-hour security-constrained economic dispatch of a
{n_buses}-bus transmission system in PyPSA, exporting the solved network.

Topology: buses named `bus0..bus{n_buses - 1}` (v_nom=380 kV) connected by the
undirected edge list {edge_str} -- implement each undirected pair as ONE
directed line from lower to higher index. Every line: reactance x={x_line} ohm
(equivalent per-unit-consistent value, resistance negligible),
s_nom = {s_nom_base} * (1 + {s_nom_jitter} * ((i * 7) % 5) / 5) MVA for the
i-th line in the edge list order, s_max_pu=1.0.

Conventional generation (marginal-cost ordered):
- `gas` at bus{gen_bus}: p_nom={gas_pnom} MW, marginal_cost={gas_mc} EUR/MWh,
  ramp_limit_up=ramp_limit_down={gas_ramp}.
- `coal` at bus{coal_bus}: p_nom={coal_pnom} MW, marginal_cost={coal_mc}
  EUR/MWh, ramp_limit_up=ramp_limit_down={coal_ramp}.

Variable renewables (p_max_pu given per hour h=0..23 by these exact formulas;
clip results into [0, 1]):
- `wind_farm` at bus{wind_bus}: p_nom={wind_pnom} MW,
  p_max_pu(h) = {wind_base} + {wind_amp} * sin(2*pi*(h - {wind_peak_h}) / 24)
                + {wind_ripple} * sin(2*pi*h / 8).
- `solar_pv` at bus{solar_bus}: p_nom={solar_pnom} MW,
  p_max_pu(h) = {solar_amp} * max(0, cos(pi * (h - 12) / 6)) ** 2.

Storage:
- `battery` at bus{bat_bus}: StorageUnit with p_nom={bat_pnom} MW,
  max_hours={bat_hours}, efficiency_store=efficiency_dispatch={bat_eff},
  standing_loss=0, ramp_limit_up=ramp_limit_down={bat_ramp},
  state_of_charge_initial = 0.4 * p_nom * max_hours MWh, cyclic operation
  (SOC returns to its initial value within solver tolerance).

Load: bus i carries constant hourly demand
p_load(i) = {load_lo} + ({load_hi} - {load_lo}) * ((i * {load_prime}) % 13) / 13
MW, modulated by a shared shape factor (1 + {load_shape_amp} *
sin(2*pi*h / 24 - pi/3)) applied identically at every bus.

Solve with `n.optimize()` (hourly snapshot weightings) and export with
`n.export_to_netcdf("network_state.nc")`.
"""
        requirements = [
            "Nodal active-power balance holds at every bus/snapshot within 1e-4 MW.",
            "No line exceeds 100% of its thermal limit.",
            "0 <= P_gen <= p_nom * p_max_pu(t) for every generator and snapshot.",
            "Generator ramps respect the stated ramp limits.",
            "Battery SOC recursion holds including converter efficiencies and ramp limits.",
        ]
        params: Dict[str, float] = {
            "grid_n_buses": float(n_buses),
            "grid_n_lines": float(len(edges)),
            "grid_s_nom_base_mva": s_nom_base,
            "grid_gas_pnom_mw": gas_pnom,
            "grid_coal_pnom_mw": coal_pnom,
            "grid_wind_pnom_mw": wind_pnom,
            "grid_solar_pnom_mw": solar_pnom,
            "grid_battery_pnom_mw": bat_pnom,
            "grid_battery_eff": bat_eff,
        }
        return SynthTaskSpec(
            id=f"grid_synth_{index:04d}",
            domain="grid",
            title=f"Economic dispatch on a synthesized {n_buses}-bus system",
            description=description,
            requirements=requirements,
            artifact_filename="network_state.nc",
            difficulty=difficulty,
            tags=["pypsa", "power-systems", "synthetic"],
            params=params,
            oracle_kwargs={},
        )

    # ------------------------------------------------------------------ #
    # Domain B: climate / ocean dynamics                                 #
    # ------------------------------------------------------------------ #

    def _build_climate_spec(self, index: int) -> SynthTaskSpec:
        rng = self._rng
        is_3d = bool(rng.random() < 0.2)
        nx = int(rng.choice([32, 48, 64, 96, 128]))
        ny = int(rng.choice([32, 48, 64, 96]))
        nz = int(rng.choice([8, 16])) if is_3d else 0
        side = float(rng.choice([5.0e5, 1.0e6, 2.0e6]))
        depth = float(rng.choice([1000.0, 2000.0]))

        dx = side / nx
        psi0 = _r(rng.uniform(2.0e4, 1.5e5), 1)             # streamfunction scale
        # Worst-case wind SPEED is sqrt(u^2 + v^2) <= sqrt(2) * pi*psi0/L;
        # size dt against that so the oracle's speed-magnitude CFL holds.
        u_max_est = np.pi * psi0 / side * float(np.sqrt(2.0))
        cfl_target = _r(rng.uniform(0.30, 0.80), 3)
        dt_raw = cfl_target * dx / u_max_est
        dt = _r(max(30.0, round(dt_raw / 30.0) * 30.0), 1)   # snap to 30 s grid
        nt = int(rng.choice([100, 150, 200, 300]))

        kappa_want = _r(rng.uniform(0.0, 60.0), 2)
        kappa_cap = 0.2 * dx * dx / dt                       # explicit stability cap
        kappa = _r(min(kappa_want, kappa_cap), 2)

        sigma_frac = _r(rng.uniform(0.06, 0.14), 3)
        blob_x = _r(rng.uniform(0.2, 0.35), 3)
        blob_y = _r(rng.uniform(0.2, 0.35), 3)

        difficulty = (
            "easy" if (nx <= 48 and nt <= 150 and not is_3d)
            else "hard" if (nx >= 96 or is_3d) else "medium"
        )
        dims_txt = (
            f"3D with vertical levels z (nz={nz}, uniform thickness "
            f"dz={_r(depth / nz, 2)} m, domain depth {depth:.0f} m)"
            if is_3d
            else "2D"
        )
        z_block = (
            "\n- coordinates `x`, `y`, `z` in meters (units attr {'units': 'm'})\n"
            if is_3d
            else "\n- coordinates `x`, `y` in meters (attrs {'units': 'm'})\n"
        )
        diff_block = (
            f"\nAfter each advection step apply explicit diffusion:\n"
            f"c <- c + kappa * dt * laplacian(c), kappa={kappa} m^2/s\n"
            f"(stable since kappa*dt/dx^2 = {_r(kappa * dt / dx / dx, 4)} <= 0.2;\n"
            f"use the discrete 5-point Laplacian, periodic)."
            if kappa > 0
            else "\nNo diffusion term is required (kappa = 0)."
        )

        description = f"""\
Simulate {dims_txt} passive-tracer advection{'-diffusion' if kappa > 0 else ''} on
a periodic square domain and export the trajectory as NetCDF.

Domain: side L={side:.0f} m, grid {nx} x {ny}{f' x {nz}' if is_3d else ''}
(dx = dy = {_r(dx, 3)} m). Wind: steady vortex streamfunction
psi(x,y) = {psi0} * sin(pi*x/L) * sin(pi*y/L) m^2/s with
u = -d(psi)/dy, v = +d(psi)/dx (centered differences suffice).
Time: dt = {dt} s for nt = {nt} steps (save all nt+1 frames). Before
integrating verify C = max|u| * dt / dx <= 1 (design target C ~= {cfl_target});
fail loudly rather than silently degrading if violated.

Initial condition: Gaussian blob centered at ({blob_x}*L, {blob_y}*L) with
sigma = {sigma_frac}*L, amplitude 1.0.{diff_block}

Numerics: flux-form conservative advection (first-order upwind acceptable;
MPDATA/MacCormack preferred) with PERIODIC boundary conditions in x and y --
the global tracer mass must be conserved to |dM/M| <= 1e-5 between the first
and last frame.

Export contract (exact names):
- data variable `tracer` with dims (time, {'z, ' if is_3d else ''}y, x),
  non-negative everywhere;
- wind components `u`, `v` (dims ({'z, ' if is_3d else ''}y, x));
{z_block}- numeric coordinate `time` in seconds spanning 0..nt*dt;
written via ds.to_netcdf("simulation_state.nc").
"""
        requirements = [
            "Global tracer mass conserved to |dM/M| <= 1e-5 between first and last frame.",
            "Tracer stays non-negative throughout (no undershoot below zero).",
            f"CFL number C = u_max*dt/dx <= 1.0 with dt={dt} s, dx={_r(dx, 3)} m.",
            "Periodic boundaries handled without mass leakage.",
            "Exact variable/coordinate names per the export contract.",
        ]
        params = {
            "clim_is_3d": 1.0 if is_3d else 0.0,
            "clim_nx": float(nx),
            "clim_ny": float(ny),
            "clim_nz": float(nz),
            "clim_side_m": side,
            "clim_dx_m": _r(dx, 4),
            "clim_dt_s": dt,
            "clim_nt": float(nt),
            "clim_psi0": psi0,
            "clim_cfl_target": cfl_target,
            "clim_kappa": kappa,
        }
        return SynthTaskSpec(
            id=f"climate_synth_{index:04d}",
            domain="climate",
            title=(
                f"Conservative {'3D ' if is_3d else '2D'}tracer transport on a "
                f"{nx}x{ny} vortex flow"
            ),
            description=description,
            requirements=requirements,
            artifact_filename="simulation_state.nc",
            difficulty=difficulty,
            tags=["xarray", "climate", "advection", "synthetic"],
            params=params,
            oracle_kwargs={"dt_seconds": dt, "cfl_max": 1.0, "tracer_mass_rtol": 1e-5},
        )

    # ------------------------------------------------------------------ #
    # Domain C: carbon kinetics                                          #
    # ------------------------------------------------------------------ #

    def _build_kinetics_spec(self, index: int) -> SynthTaskSpec:
        rng = self._rng
        ptsa = bool(rng.random() < 0.35)
        n_cycles = int(rng.integers(4, 9))
        t_ads_h = _r(rng.uniform(4.0, 8.0), 2)
        t_des_h = _r(rng.uniform(1.5, 3.0), 2)
        t_ads_k = _r(rng.uniform(293.0, 303.0), 1)
        t_des_k = _r(rng.uniform(373.0, 423.0), 1)
        p_feed_bar = _r(rng.uniform(2.0e-4, 6.0e-4), 6)
        p_des_bar = _r(rng.uniform(0.01, 0.05), 4) if ptsa else 1.0

        has_preheat = bool(rng.random() < 0.4)
        t_mid_k = _r((t_ads_k + t_des_k) / 2.0, 1)
        t_pre_h = _r(rng.uniform(0.5, 1.5), 2)

        q_max = _r(rng.uniform(2.0, 5.0), 2)                 # mol/kg
        b0 = _r(rng.uniform(1.0e-6, 1.0e-4), 10)             # 1/bar at T_ref
        dh_ads = _r(rng.uniform(60.0e3, 110.0e3), 1)         # J/mol (positive mag.)
        t_ref_k = 298.15
        ea = _r(rng.uniform(15.0e3, 45.0e3), 1)              # J/mol
        k_ref = _r(rng.uniform(8.0e-4, 4.0e-3), 7)           # 1/s at T_ref
        dt_int = int(rng.choice([5.0, 10.0, 15.0]))
        ss_rtol = _r(rng.uniform(2.0e-3, 1.0e-2), 4)

        min_cycles = max(3, n_cycles - 2)
        difficulty = (
            "hard" if (ptsa or has_preheat or n_cycles >= 7) else
            "easy" if (n_cycles <= 5 and not has_preheat) else "medium"
        )
        stage_txt = (
            f"- Preheat stage: {t_pre_h} h at T={t_mid_k} K (pressure held at feed level);\n"
            if has_preheat
            else ""
        )
        des_pressure_txt = (
            f"under vacuum at P={p_des_bar} bar" if ptsa else "under inert purge at ~0 bar gauge"
        )
        swing_name = "pressure-and-temperature-swing (PTSA)" if ptsa else \
            "pure temperature-swing (TSA)"

        description = f"""\
Simulate a solid-sorbent direct-air-capture bed operating repeated multistage
{swing_name} cycles until cyclic steady state; export the full time series.

Cycle recipe (repeat n_cycles={n_cycles} times, in this stage order):
- Adsorption stage: {t_ads_h} h at T={t_ads_k} K with feed p_co2={p_feed_bar} bar.
{stage_txt}- Desorption stage: {t_des_h} h at T={t_des_k} K, {des_pressure_txt};
  ALL CO2 released during this stage counts as captured.

Equilibrium (Langmuir with van't Hoff temperature dependence):
q_eq(T, p) = q_max * (b(T) * p) / (1 + b(T) * p)      [mol/kg]
b(T) = {b0} * exp(({dh_ads} / R) * (1/T - 1/{t_ref_k}))  [1/bar], R = {_R_GAS}
q_max = {q_max} mol/kg. Note b(T) DECREASES with temperature (exothermic
adsorption), which is what makes the swing regenerative.

Kinetics: linear driving force dm/dt = k(T) * (q_eq(T,p) - m),
k(T) = {k_ref} * exp(-{ea}/R * (1/T - 1/{t_ref_k}))  [1/s].
Integrate with explicit Euler using dt <= {dt_int:.0f} s.

Bookkeeping (all on a common `time` dimension):
- `cycle`: integer running cycle index per timestep;
- `m_adsorbed`: instantaneous bed loading [kg], never negative;
- `captured_cumulative`: monotone non-decreasing cumulative captured mass
  [kg]; increments ONLY during desorption stages;
- `T_bed`: bed temperature [K];
- `p_bed`: bed CO2 partial pressure [bar] following the stage schedule.

Export with ds.to_netcdf("dac_state.nc"). The verifier checks per-cycle
capture balance (desorbed bed mass == captured increment) and cycle-to-cycle
drift (cyclic steady state, rtol={ss_rtol}).
"""
        requirements = [
            "Per-cycle mass balance: desorbed bed mass equals captured_cumulative increment.",
            "Bed loading remains physically bounded (>= 0) at every timestep.",
            f"Cyclic steady state reached: cycle-to-cycle drift <= {ss_rtol}.",
            "Temperature stays positive and follows the staged schedule approximately.",
            f"At least {min_cycles} complete cycles recorded.",
        ]
        params = {
            "kin_ptsa": 1.0 if ptsa else 0.0,
            "kin_has_preheat": 1.0 if has_preheat else 0.0,
            "kin_n_cycles": float(n_cycles),
            "kin_t_ads_h": t_ads_h,
            "kin_t_des_h": t_des_h,
            "kin_t_ads_k": t_ads_k,
            "kin_t_des_k": t_des_k,
            "kin_p_feed_bar": p_feed_bar,
            "kin_q_max": q_max,
            "kin_b0": b0,
            "kin_dh_ads": dh_ads,
            "kin_ea": ea,
            "kin_k_ref": k_ref,
            "kin_ss_rtol": ss_rtol,
        }
        return SynthTaskSpec(
            id=f"kinetics_synth_{index:04d}",
            domain="kinetics",
            title=(
                f"Multistage {'PTSA' if ptsa else 'TSA'} DAC cycle "
                f"({n_cycles} cycles) to steady state"
            ),
            description=description,
            requirements=requirements,
            artifact_filename="dac_state.nc",
            difficulty=difficulty,
            tags=["kinetics", "dac", "adsorption", "synthetic"],
            params=params,
            oracle_kwargs={
                "steady_state_rtol": ss_rtol,
                "min_cycles": min_cycles,
                "balance_rtol": 1e-4,
            },
        )


if __name__ == "__main__":  # pragma: no cover - CLI convenience
    parser = argparse.ArgumentParser(description="Generate benchmark_suite.jsonl")
    parser.add_argument("--total", type=int, default=600)
    parser.add_argument("--per-domain", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("-o", "--out", default=str(DEFAULT_SUITE_PATH))
    args = parser.parse_args()
    _synth = PhysicsTaskSynthesizer(seed=args.seed)
    _specs = _synth.synthesize(total=args.total, per_domain=args.per_domain)
    _path = PhysicsTaskSynthesizer.export_jsonl(_specs, args.out)
    _by_domain: Dict[str, int] = {}
    for _s in _specs:
        _by_domain[_s.domain] = _by_domain.get(_s.domain, 0) + 1
    print(f"wrote {len(_specs)} tasks -> {_path} ({_by_domain})")
