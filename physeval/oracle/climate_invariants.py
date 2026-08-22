"""Physical-conservation oracle for gridded climate/ocean simulation states.

:class:`ClimateGridOracle` loads an :class:`xarray.Dataset` (NetCDF ``.nc``,
or Zarr ``.zarr`` directory store) exported by sandbox code and asserts:

1. First-law global tracer mass conservation:
   ``|(M_final - M_initial) / M_initial| <= tracer_mass_rtol``.
2. Physical quantity bounds: density ``rho > 0``, specific humidity
   ``q >= 0``, absolute temperature ``T > 0 K``.
3. Courant-Friedrichs-Lewy (CFL) numerical stability:
   ``C = |u_max| * dt / dx <= 1.0``.

Variable discovery is name-based with configurable candidate lists, so the
oracle tolerates heterogeneous model naming conventions while remaining
deterministic (first match in candidate order always wins).
"""

from __future__ import annotations

import math
import os
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Tuple

import numpy as np

from physeval.oracle.base import (
    BasePhysicsOracle,
    StateFileError,
    VerificationResult,
)

__all__ = ["ClimateGridOracle"]

#: Earth mean radius in meters (WGS84 mean).
_EARTH_RADIUS_M = 6.371e6


def _open_dataset(state_file_path: str) -> Any:
    """Open *state_file_path* as an xarray Dataset (lazy imports kept local)."""
    try:
        import xarray as xr
    except ImportError as exc:
        raise StateFileError(
            "Optional dependency 'xarray' is required to verify climate states. "
            "Install with: pip install 'physeval-agent[climate]'"
        ) from exc
    try:
        if state_file_path.endswith(".zarr"):
            return xr.open_zarr(state_file_path, consolidated=False)
        return xr.open_dataset(state_file_path)
    except Exception as exc:
        raise StateFileError(
            f"Failed to open climate dataset at {state_file_path!r}: {exc}"
        ) from exc


def xr_align_weights(w_da: Any, target: Any, dims: Sequence[str]) -> Any:
    """Broadcast a weight array against *target* over *dims*.

    Thin wrapper kept module-level so it can be unit-tested without a full
    oracle instance.
    """
    return w_da.broadcast_like(target[list(dims)])


class ClimateGridOracle(BasePhysicsOracle):
    """Deterministic verifier for gridded atmospheric/oceanic states."""

    name: ClassVar[str] = "climate-grid-invariants"
    supported_extensions: ClassVar[Tuple[str, ...]] = (".nc", ".nc4", ".cdf", ".zarr")

    def __init__(
        self,
        *,
        tracer_mass_rtol: float = 1e-5,
        cfl_max: float = 1.0,
        dt_seconds: Optional[float] = None,
        dx_meters: Optional[float] = None,
        tracer_candidates: Sequence[str] = (
            "tracer",
            "c",
            "conc",
            "concentration",
            "passive_tracer",
            "q_tracer",
            "co2",
        ),
        density_candidates: Sequence[str] = ("rho", "density", "air_density"),
        humidity_candidates: Sequence[str] = (
            "q",
            "qv",
            "specific_humidity",
            "hus",
            "sphu",
        ),
        temperature_candidates: Sequence[str] = (
            "T",
            "temp",
            "temperature",
            "tas",
            "t2m",
            "air_temperature",
            "theta",
        ),
        u_wind_candidates: Sequence[str] = ("u", "ua", "uas", "u10", "wind_u"),
        v_wind_candidates: Sequence[str] = ("v", "va", "vas", "v10", "wind_v"),
        area_candidates: Sequence[str] = (
            "cell_area",
            "area",
            "areacella",
            "areacello",
            "dx_dy",
        ),
    ) -> None:
        """Configure tolerances and variable-name candidates.

        Args:
            tracer_mass_rtol: Allowed relative drift of global tracer mass.
            cfl_max: Maximum admissible Courant number (spec: 1.0).
            dt_seconds: Explicit timestep override; inferred from the time
                coordinate when omitted.
            dx_meters: Explicit grid-spacing override; derived from
                coordinates when omitted.
            tracer_candidates: Variable names probed for the conserved tracer.
            density_candidates: Names probed for density fields.
            humidity_candidates: Names probed for specific-humidity fields.
            temperature_candidates: Names probed for absolute temperature.
            u_wind_candidates / v_wind_candidates: Wind component names used
                in the CFL estimate.
            area_candidates: Cell-area variable names used to weight the
                global mass integral; uniform weights are used if absent.
        """
        if tracer_mass_rtol < 0 or cfl_max <= 0:
            raise ValueError("tracer_mass_rtol must be >= 0 and cfl_max > 0.")
        self.tracer_mass_rtol = float(tracer_mass_rtol)
        self.cfl_max = float(cfl_max)
        self.dt_seconds = float(dt_seconds) if dt_seconds is not None else None
        self.dx_meters = float(dx_meters) if dx_meters is not None else None
        self.tracer_candidates = tuple(tracer_candidates)
        self.density_candidates = tuple(density_candidates)
        self.humidity_candidates = tuple(humidity_candidates)
        self.temperature_candidates = tuple(temperature_candidates)
        self.u_wind_candidates = tuple(u_wind_candidates)
        self.v_wind_candidates = tuple(v_wind_candidates)
        self.area_candidates = tuple(area_candidates)

    # ------------------------------------------------------------------ #
    # Entry point                                                        #
    # ------------------------------------------------------------------ #

    def verify(self, state_file_path: str) -> VerificationResult:
        """Run conservation, bounds and CFL checks on the dataset."""
        # Zarr stores are directories, so bypass the file-only pre-check.
        if not state_file_path.endswith(".zarr"):
            self.validate_path(state_file_path)
        elif not os.path.isdir(state_file_path):
            raise StateFileError(f"Zarr store does not exist: {state_file_path!r}")

        ds = _open_dataset(state_file_path)
        violations: List[Any] = []
        num_steps = 0.0
        for dim_size in ds.sizes.values():
            num_steps = float(dim_size)
            break
        metrics: Dict[str, float] = {"num_timesteps": num_steps}
        try:
            for checker in (
                self._check_tracer_mass_conservation,
                self._check_physical_bounds,
                self._check_cfl_stability,
            ):
                try:
                    checker(ds, violations, metrics)
                except Exception as exc:
                    violations.append(
                        self.make_violation(
                            name=f"internal_error:{checker.__name__}",
                            severity="FATAL",
                            observed=float("nan"),
                            threshold=float("nan"),
                            message=f"Oracle subsystem {checker.__name__} crashed: {exc}",
                        )
                    )
        finally:
            ds.close()

        has_fatal = any(v.severity == "FATAL" for v in violations)
        return VerificationResult(
            passed=not has_fatal,
            violations=violations,
            metrics=metrics,
        )

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #

    def _find_var(self, ds: Any, candidates: Sequence[str]) -> Optional[str]:
        """Return the first data variable matching a candidate name."""
        variables = set(map(str, ds.data_vars))
        for cand in candidates:
            if cand in variables:
                return cand
        lowered = {v.lower(): v for v in variables}
        for cand in candidates:
            hit = lowered.get(cand.lower())
            if hit is not None:
                return hit
        return None

    @staticmethod
    def _finite_min(da: Any) -> float:
        """NaN-aware minimum of an array as a plain float."""
        vals = np.asarray(da.values, dtype=float)
        finite = vals[np.isfinite(vals)]
        return float(finite.min()) if finite.size else float("nan")

    @staticmethod
    def _finite_max(da: Any) -> float:
        """NaN-aware maximum of an array as a plain float."""
        vals = np.asarray(da.values, dtype=float)
        finite = vals[np.isfinite(vals)]
        return float(finite.max()) if finite.size else float("nan")

    def _time_dim(self, da: Any) -> str:
        """Heuristic time dimension of a data array."""
        for dim in ("time", "t", "step"):
            if dim in da.dims:
                return dim
        return str(da.dims[0])

    # ------------------------------------------------------------------ #
    # Check 1: first-law global tracer mass conservation                 #
    # ------------------------------------------------------------------ #

    def _check_tracer_mass_conservation(
        self, ds: Any, violations: List[Any], metrics: Dict[str, float]
    ) -> None:
        """Assert |(M_final - M_initial)/M_initial| <= tracer_mass_rtol."""
        var = self._find_var(ds, self.tracer_candidates)
        if var is None:
            violations.append(
                self.make_violation(
                    name="tracer_variable_missing",
                    severity="FATAL",
                    observed=0.0,
                    threshold=0.0,
                    message=(
                        f"No tracer variable found (probed {list(self.tracer_candidates)}); "
                        f"data variables present: {sorted(map(str, ds.data_vars))}."
                    ),
                )
            )
            return

        da = ds[var]
        tdim = self._time_dim(da)
        spatial_dims = [d for d in da.dims if d != tdim]

        weights = None
        area_var = self._find_var(ds, self.area_candidates)
        if area_var is not None:
            w_da = ds[area_var]
            shared = [d for d in spatial_dims if d in w_da.dims]
            if set(shared) == set(w_da.dims):
                weights = xr_align_weights(w_da, da, shared)

        filled = da.fillna(0.0)
        if weights is not None:
            mass_series = (filled * weights).sum(dim=spatial_dims)
        else:
            mass_series = filled.sum(dim=spatial_dims)

        m_values = np.asarray(mass_series.values, dtype=float).ravel()
        if m_values.size == 0:
            violations.append(
                self.make_violation(
                    name="tracer_mass_conservation",
                    severity="FATAL",
                    observed=0.0,
                    threshold=self.tracer_mass_rtol,
                    message=f"Tracer '{var}' has zero timesteps; cannot verify conservation.",
                )
            )
            return

        m_initial, m_final = float(m_values[0]), float(m_values[-1])
        scale = max(abs(m_initial), abs(m_final), 1e-30)
        drift = abs(m_final - m_initial) / scale
        metrics["tracer_mass_initial"] = m_initial
        metrics["tracer_mass_final"] = m_final
        metrics["tracer_relative_drift"] = drift

        if drift > self.tracer_mass_rtol:
            violations.append(
                self.make_violation(
                    name="tracer_mass_conservation",
                    severity="FATAL",
                    observed=drift,
                    threshold=self.tracer_mass_rtol,
                    message=(
                        f"Global tracer mass not conserved for '{var}': "
                        f"M_initial={m_initial:.8g}, M_final={m_final:.8g}, "
                        f"|drift/M|={drift:.3e} > rtol={self.tracer_mass_rtol:.1e}. "
                        "Check advection scheme flux form and boundary conditions."
                    ),
                )
            )

    # ------------------------------------------------------------------ #
    # Check 2: physical quantity bounds                                  #
    # ------------------------------------------------------------------ #

    def _check_physical_bounds(
        self, ds: Any, violations: List[Any], metrics: Dict[str, float]
    ) -> None:
        """Assert rho > 0, q >= 0, T > 0 K over every grid cell."""
        # mode "gt": strictly greater than floor (rho > 0, T > 0 K);
        # mode "ge": greater-or-equal with tiny numerical slack (q >= 0).
        checks: Tuple[Tuple[Sequence[str], str, str, float, str], ...] = (
            (self.density_candidates, "density_positive", "Density", 0.0, "gt"),
            (self.humidity_candidates, "humidity_nonnegative", "Specific humidity", 0.0, "ge"),
            (self.temperature_candidates, "temperature_positive", "Temperature", 0.0, "gt"),
        )
        for candidates, invariant, label, floor, mode in checks:
            var = self._find_var(ds, candidates)
            if var is None:
                continue
            da = ds[var]
            vmin = self._finite_min(da)
            metrics[f"{var}_min"] = vmin
            metrics[f"{var}_max"] = self._finite_max(da)
            if math.isnan(vmin):
                continue
            breached = (vmin <= floor) if mode == "gt" else (vmin < floor - 1e-12)
            operator = "> 0" if mode == "gt" else ">= 0"
            if breached:
                violations.append(
                    self.make_violation(
                        name=invariant,
                        severity="FATAL",
                        observed=vmin,
                        threshold=floor,
                        message=(
                            f"{label} variable '{var}' violates physical bounds: "
                            f"min={vmin:.6g} must be {operator}."
                        ),
                    )
                )

        temp_var = self._find_var(ds, self.temperature_candidates)
        if temp_var is not None:
            vmax = self._finite_max(ds[temp_var])
            vmin = metrics.get(f"{temp_var}_min", float("nan"))
            plausible_lo, plausible_hi = 50.0, 500.0
            if not math.isnan(vmin) and (vmin < plausible_lo or vmax > plausible_hi):
                violations.append(
                    self.make_violation(
                        name="temperature_implausible_range",
                        severity="WARNING",
                        observed=float(vmin),
                        threshold=plausible_lo,
                        message=(
                            f"Temperature '{temp_var}' outside plausible Earth range "
                            f"[{plausible_lo:.0f} K, {plausible_hi:.0f} K]: "
                            f"min={vmin:.4g} K, max={vmax:.4g} K."
                        ),
                    )
                )

    # ------------------------------------------------------------------ #
    # Check 3: Courant-Friedrichs-Lewy numerical stability               #
    # ------------------------------------------------------------------ #

    def _infer_dt_seconds(self, ds: Any) -> Optional[float]:
        """Explicit override, else mean spacing of a time-like coordinate."""
        if self.dt_seconds is not None:
            return self.dt_seconds
        for name in ("time", "t", "step"):
            if name not in ds.coords:
                continue
            coord = ds[name]
            values = np.asarray(coord.values)
            if values.size < 2:
                return None
            if np.issubdtype(values.dtype, np.datetime64):
                deltas = np.diff(values).astype("timedelta64[s]").astype(float)
                positive = deltas[deltas > 0]
                return float(positive.mean()) if positive.size else None
            numeric = np.abs(np.diff(values.astype(float)))
            positive = numeric[numeric > 0]
            if not positive.size:
                continue
            units = str(coord.attrs.get("units", "")).lower()
            factor = 1.0
            for token, mult in (("day", 86400.0), ("hour", 3600.0), ("hr", 3600.0),
                                ("minute", 60.0), ("min", 60.0)):
                if token in units:
                    factor = mult
                    break
            else:
                if units.endswith("d") or " d" in units:
                    factor = 86400.0
            return float(positive.mean() * factor)
        return None

    def _infer_dx_meters(self, ds: Any) -> Optional[float]:
        """Explicit override, else great-circle spacing from lon/lat or x."""
        if self.dx_meters is not None:
            return self.dx_meters
        if "x" in ds.coords:
            xv = np.asarray(ds["x"].values, dtype=float)
            if xv.size >= 2:
                step = float(np.nanmean(np.abs(np.diff(xv))))
                units = str(ds["x"].attrs.get("units", "m")).lower()
                if "deg" in units:
                    return math.radians(step) * _EARTH_RADIUS_M
                return step
        if "lon" in ds.coords and "lat" in ds.coords:
            lat = np.asarray(ds["lat"].values, dtype=float)
            lon = np.asarray(ds["lon"].values, dtype=float)
            if lon.size >= 2:
                mean_lat = math.radians(float(np.nanmean(np.abs(lat)))) if lat.size else 0.0
                dlon = float(np.nanmean(np.abs(np.diff(lon))))
                return math.radians(dlon) * _EARTH_RADIUS_M * max(math.cos(mean_lat), 1e-3)
        return None

    def _check_cfl_stability(
        self, ds: Any, violations: List[Any], metrics: Dict[str, float]
    ) -> None:
        """Assert C = u_max * dt / dx <= cfl_max."""
        u_var = self._find_var(ds, self.u_wind_candidates)
        v_var = self._find_var(ds, self.v_wind_candidates)
        dt = self._infer_dt_seconds(ds)
        dx = self._infer_dx_meters(ds)

        missing: List[str] = []
        if u_var is None:
            missing.append("zonal wind u")
        if dt is None:
            missing.append("timestep dt")
        if dx is None:
            missing.append("grid spacing dx")

        if u_var is None or dt is None or dx is None or dx <= 0:
            violations.append(
                self.make_violation(
                    name="cfl_not_evaluable",
                    severity="WARNING",
                    observed=0.0,
                    threshold=self.cfl_max,
                    message=(
                        "CFL check skipped; could not determine: "
                        + ", ".join(missing)
                        + f". Available vars: {sorted(map(str, ds.data_vars))}."
                    ),
                )
            )
            return

        u = np.abs(np.asarray(ds[u_var].values, dtype=float))
        speed_max = float(np.nanmax(u)) if u.size else 0.0
        if v_var is not None:
            v = np.asarray(ds[v_var].values, dtype=float)
            combined = np.sqrt(u**2 + np.asarray(v, dtype=float) ** 2)
            speed_max = float(np.nanmax(combined)) if combined.size else speed_max

        courant = speed_max * dt / dx
        metrics["cfl_u_max_ms"] = speed_max
        metrics["cfl_dt_s"] = dt
        metrics["cfl_dx_m"] = dx
        metrics["max_courant_number"] = courant

        if courant > self.cfl_max:
            violations.append(
                self.make_violation(
                    name="cfl_stability",
                    severity="FATAL",
                    observed=courant,
                    threshold=self.cfl_max,
                    message=(
                        f"CFL condition violated: C=u_max*dt/dx={courant:.4f} > "
                        f"{self.cfl_max} (u_max={speed_max:.4g} m/s, dt={dt:.4g} s, "
                        f"dx={dx:.4g} m). Reduce timestep or increase grid spacing."
                    ),
                )
            )

