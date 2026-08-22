"""Physical-feasibility oracle for serialized PyPSA transmission networks.

:class:`PyPSAGridOracle` deserializes an optimized/solved PyPSA network
(``*.nc`` NetCDF export or ``*.h5`` HDF5 store) written by sandbox code and
asserts four families of physical invariants:

1. Nodal active-power balance (Kirchhoff's current law): for every bus and
   snapshot the signed sum of component injections plus branch flows must
   vanish within ``balance_tol_mw``.
2. Line / transformer thermal limits: ``|P_ij| <= s_nom * s_max_pu``.
3. Generator capacity limits: ``0 <= P_g <= p_nom * p_max_pu(t)``.
4. Storage energy conservation: state-of-charge trajectories must satisfy the
   discrete SOC recursion including converter efficiencies, standing losses,
   inflow and spillage.

All checks are pure functions of the state file, making verdicts fully
deterministic and replayable.
"""

from __future__ import annotations

from typing import Any, ClassVar, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from physeval.oracle.base import (
    BasePhysicsOracle,
    StateFileError,
    VerificationResult,
)

__all__ = ["PyPSAGridOracle"]


def _load_network(state_file_path: str) -> Any:
    """Deserialize a PyPSA network from ``.nc`` or ``.h5``.

    The import is deliberately deferred so that ``physeval`` remains
    importable on machines without PyPSA installed.

    Raises:
        StateFileError: If PyPSA is unavailable or deserialization fails.
    """
    try:
        import pypsa
    except ImportError as exc:
        raise StateFileError(
            "Optional dependency 'pypsa' is required to verify grid states. "
            "Install with: pip install 'physeval-agent[grid]'"
        ) from exc
    try:
        return pypsa.Network(state_file_path)
    except Exception as exc:
        raise StateFileError(
            f"Failed to deserialize PyPSA network from {state_file_path!r}: {exc}"
        ) from exc


def _component_timeseries(net: Any, component: str, attr: str) -> Optional[pd.DataFrame]:
    """Return the dynamic time series *attr* of *component* if populated.

    Args:
        net: A loaded :class:`pypsa.Network`.
        component: Plural component name, e.g. ``"generators"``.
        attr: Time-series attribute, e.g. ``"p"``.

    Returns:
        The dynamic DataFrame, or ``None`` when absent/empty.
    """
    container = getattr(net, f"{component}_t", None)
    ts = getattr(container, attr, None)
    if isinstance(ts, pd.DataFrame) and not ts.empty:
        return ts
    return None


def _group_by_bus(frame: pd.DataFrame, bus_of: pd.Series) -> pd.DataFrame:
    """Collapse component-indexed columns onto their buses.

    Args:
        frame: Snapshots x components frame.
        bus_of: Mapping from component name to bus id.

    Returns:
        Snapshots x buses frame of aggregated magnitudes.
    """
    cols = [c for c in frame.columns if c in bus_of.index]
    if not cols:
        return pd.DataFrame(0.0, index=frame.index, columns=[])
    grouped = frame[cols].T.groupby(bus_of.loc[cols]).sum().T
    return grouped


class PyPSAGridOracle(BasePhysicsOracle):
    """Deterministic verifier for PyPSA network states.

    Example:
        >>> oracle = PyPSAGridOracle()
        >>> result = oracle.verify("runs/exp01/network_state.nc")  # doctest: +SKIP
        >>> result.passed
        True
    """

    name: ClassVar[str] = "pypsa-grid-invariants"
    supported_extensions: ClassVar[Tuple[str, ...]] = (".nc", ".h5", ".h5store")

    def __init__(
        self,
        *,
        balance_tol_mw: float = 1e-4,
        line_loading_tol: float = 1e-6,
        capacity_tol_mw: float = 1e-6,
        soc_atol_mwh: float = 1e-3,
        soc_rtol: float = 1e-4,
    ) -> None:
        """Configure tolerances.

        Args:
            balance_tol_mw: Max absolute nodal imbalance (Kirchhoff).
            line_loading_tol: Relative slack allowed above 100% loading.
            capacity_tol_mw: Absolute slack allowed on generator bounds.
            soc_atol_mwh: Absolute SOC-residual tolerance.
            soc_rtol: Relative SOC-residual tolerance (scaled by peak SOC).
        """
        if balance_tol_mw < 0 or capacity_tol_mw < 0 or soc_atol_mwh < 0:
            raise ValueError("Absolute tolerances must be non-negative.")
        self.balance_tol_mw = float(balance_tol_mw)
        self.line_loading_tol = float(line_loading_tol)
        self.capacity_tol_mw = float(capacity_tol_mw)
        self.soc_atol_mwh = float(soc_atol_mwh)
        self.soc_rtol = float(soc_rtol)

    # ------------------------------------------------------------------ #
    # Entry point                                                        #
    # ------------------------------------------------------------------ #

    def verify(self, state_file_path: str) -> VerificationResult:
        """Run all four invariant families against *state_file_path*."""
        self.validate_path(state_file_path)
        net = _load_network(state_file_path)

        violations: List[Any] = []
        metrics: Dict[str, float] = {
            "num_snapshots": float(len(getattr(net, "snapshots", []))),
            "num_buses": float(len(getattr(net, "buses", []))),
        }

        for checker in (
            self._check_nodal_balance,
            self._check_branch_thermal_limits,
            self._check_generator_capacity,
            self._check_storage_conservation,
        ):
            try:
                checker(net, violations, metrics)
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

        has_fatal = any(v.severity == "FATAL" for v in violations)
        return VerificationResult(
            passed=not has_fatal,
            violations=violations,
            metrics=metrics,
        )

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _snapshot_weights_hours(net: Any) -> pd.Series:
        """Return snapshot durations in hours (defaults to 1.0 h)."""
        sw = getattr(net, "snapshot_weightings", None)
        snapshots = net.snapshots
        if isinstance(sw, pd.Series) and len(sw) == len(snapshots):
            return sw.astype(float)
        if isinstance(sw, pd.DataFrame) and "generators" in getattr(sw, "columns", []):
            return sw["generators"].astype(float)
        return pd.Series(1.0, index=snapshots)

    # ------------------------------------------------------------------ #
    # Check 1: nodal active-power balance (Kirchhoff)                    #
    # ------------------------------------------------------------------ #

    def _check_nodal_balance(
        self,
        net: Any,
        violations: List[Any],
        metrics: Dict[str, float],
    ) -> None:
        """Assert |sum(P_in) - sum(P_out)| <= balance_tol_mw at every bus."""
        snapshots = net.snapshots
        buses = net.buses.index
        injection = pd.DataFrame(0.0, index=snapshots, columns=buses)

        def accumulate(component: str, attr: str, static_attr: str, sign: float) -> None:
            nonlocal injection
            static = getattr(net, component, None)
            if static is None or static.empty:
                return
            ts = _component_timeseries(net, component, attr)
            if ts is None:
                if static_attr not in static.columns:
                    return
                vals = pd.to_numeric(static[static_attr], errors="coerce").fillna(0.0)
                data = np.tile(vals.to_numpy(dtype=float), (len(snapshots), 1))
                ts = pd.DataFrame(data, index=snapshots, columns=vals.index)
            grouped = _group_by_bus(ts, static["bus"])
            aligned = grouped.reindex(index=snapshots, columns=buses, fill_value=0.0)
            injection = injection.add(sign * aligned, fill_value=0.0)

        accumulate("generators", "p", "p_set", +1.0)
        accumulate("loads", "p", "p_set", -1.0)
        accumulate("storage_units", "p", "p_set", +1.0)
        accumulate("stores", "p", "p_set", +1.0)

        # Branch flows: p0 > 0 means flow bus0 -> bus1 (p1 == -p0).
        for comp, attr0 in (("lines", "p0"), ("transformers", "p0")):
            static = getattr(net, comp, None)
            ts0 = _component_timeseries(net, comp, attr0)
            if static is None or static.empty or ts0 is None:
                continue
            leaving = _group_by_bus(ts0, static["bus0"])
            arriving = _group_by_bus(ts0, static["bus1"])
            injection = injection.sub(leaving.reindex(columns=buses, fill_value=0.0), fill_value=0.0)
            injection = injection.add(arriving.reindex(columns=buses, fill_value=0.0), fill_value=0.0)

        imbalance = injection.abs()
        if imbalance.empty or imbalance.size == 0 or not imbalance.to_numpy().any():
            metrics["max_nodal_imbalance_mw"] = 0.0
            return

        finite = imbalance.replace([np.inf, -np.inf], np.nan).dropna(how="all")
        if finite.empty:
            metrics["max_nodal_imbalance_mw"] = float("nan")
            violations.append(
                self.make_violation(
                    name="nodal_power_balance",
                    severity="FATAL",
                    observed=float("nan"),
                    threshold=self.balance_tol_mw,
                    message="Nodal injections contained only non-finite values.",
                )
            )
            return

        stacked = finite.stack()
        worst_idx = stacked.idxmax()
        worst = float(stacked.max())
        metrics["max_nodal_imbalance_mw"] = worst
        if worst > self.balance_tol_mw:
            violations.append(
                self.make_violation(
                    name="nodal_power_balance",
                    severity="FATAL",
                    observed=worst,
                    threshold=self.balance_tol_mw,
                    message=(
                        f"Kirchhoff violated: bus={worst_idx[1]} snapshot={worst_idx[0]} "
                        f"imbalance={worst:.6g} MW exceeds tolerance {self.balance_tol_mw:.6g} MW."
                    ),
                )
            )

    # ------------------------------------------------------------------ #
    # Check 2: branch thermal limits                                     #
    # ------------------------------------------------------------------ #

    def _check_branch_thermal_limits(
        self,
        net: Any,
        violations: List[Any],
        metrics: Dict[str, float],
    ) -> None:
        """Assert |P_ij| <= s_nom * s_max_pu for lines and transformers."""
        max_loading = 0.0
        unlimited = 0
        for comp in ("lines", "transformers"):
            static = getattr(net, comp, None)
            ts0 = _component_timeseries(net, comp, "p0")
            if static is None or static.empty or ts0 is None:
                continue
            s_nom = pd.to_numeric(static.get("s_nom"), errors="coerce")
            s_max_pu = (
                pd.to_numeric(static.get("s_max_pu"), errors="coerce").fillna(1.0)
                if "s_max_pu" in static.columns
                else pd.Series(1.0, index=static.index)
            )
            limit = (s_nom * s_max_pu).astype(float)

            constrained = limit.replace([np.inf, -np.inf], np.nan).dropna()
            unlimited += int(len(limit) - len(constrained))
            usable = [c for c in ts0.columns if c in constrained.index]
            if not usable:
                continue
            flow = ts0[usable].abs()
            lim = constrained.loc[usable]
            zero_mask = lim <= 0
            if zero_mask.any():
                violations.append(
                    self.make_violation(
                        name=f"{comp}_nonpositive_s_nom",
                        severity="WARNING",
                        observed=float(zero_mask.sum()),
                        threshold=0.0,
                        message=f"{zero_mask.sum()} {comp} have s_nom<=0 and were skipped.",
                    )
                )
                usable = [c for c in usable if not bool(zero_mask.loc[c])]
                if not usable:
                    continue
                flow = ts0[usable].abs()
                lim = constrained.loc[usable]

            loading = flow.div(lim, axis=1)
            finite_loading = loading.replace([np.inf, -np.inf], np.nan)
            stacked = finite_loading.stack()
            if stacked.empty:
                continue
            worst_idx = stacked.idxmax()
            worst_ratio = float(stacked.max())
            max_loading = max(max_loading, worst_ratio)
            if worst_ratio > 1.0 + self.line_loading_tol:
                violations.append(
                    self.make_violation(
                        name=f"{comp}_thermal_limit",
                        severity="FATAL",
                        observed=worst_ratio,
                        threshold=1.0,
                        message=(
                            f"{comp.rstrip('s').capitalize()} '{worst_idx[1]}' overloaded at "
                            f"snapshot {worst_idx[0]}: |P|={float(flow.loc[worst_idx]):.6g} MW "
                            f"> limit={float(lim.loc[worst_idx[1]]):.6g} MW "
                            f"(loading {100.0 * worst_ratio:.3f}%)."
                        ),
                    )
                )

        metrics["max_branch_loading_pct"] = 100.0 * max_loading
        metrics["branches_without_limit"] = float(unlimited)

    # ------------------------------------------------------------------ #
    # Check 3: generator capacity constraints                            #
    # ------------------------------------------------------------------ #

    def _check_generator_capacity(
        self,
        net: Any,
        violations: List[Any],
        metrics: Dict[str, float],
    ) -> None:
        """Assert 0 - tol <= P_g <= p_nom * p_max_pu(t) + tol."""
        gens = getattr(net, "generators", None)
        p = _component_timeseries(net, "generators", "p")
        if gens is None or gens.empty:
            metrics["generators_checked"] = 0.0
            return
        if p is None:
            violations.append(
                self.make_violation(
                    name="generator_dispatch_missing",
                    severity="FATAL",
                    observed=0.0,
                    threshold=0.0,
                    message=(
                        "No generator dispatch series (generators_t.p) found in state; "
                        "the network was likely never solved."
                    ),
                )
            )
            return

        p_nom = pd.to_numeric(gens.get("p_nom"), errors="coerce").fillna(0.0).astype(float)
        p_max_pu_static = (
            pd.to_numeric(gens.get("p_max_pu"), errors="coerce").fillna(1.0).astype(float)
            if "p_max_pu" in gens.columns
            else pd.Series(1.0, index=gens.index)
        )
        pu_ts = _component_timeseries(net, "generators", "p_max_pu")

        if pu_ts is None:
            upper = pd.DataFrame(
                np.outer(np.ones(len(p.index)), (p_nom * p_max_pu_static).to_numpy()),
                index=p.index,
                columns=p.columns,
            )
        else:
            upper = p.mul(pu_ts.reindex(columns=p.columns), axis=1).fillna(0.0)

        cols = [c for c in p.columns if c in gens.index]
        if not cols:
            metrics["generators_checked"] = 0.0
            return
        p_sel, up_sel = p[cols], upper[cols]
        metrics["generators_checked"] = float(len(cols))

        excess_high = (p_sel.sub(up_sel)).stack()
        excess_low = (-p_sel).stack()

        worst_high = float(excess_high.max()) if not excess_high.empty else 0.0
        worst_low = float(excess_low.max()) if not excess_low.empty else 0.0
        metrics["max_gen_above_capacity_mw"] = max(worst_high, 0.0)
        metrics["max_gen_below_zero_mw"] = max(worst_low, 0.0)

        if worst_high > self.capacity_tol_mw:
            idx = excess_high.idxmax()
            violations.append(
                self.make_violation(
                    name="generator_upper_bound",
                    severity="FATAL",
                    observed=worst_high,
                    threshold=self.capacity_tol_mw,
                    message=(
                        f"Generator '{idx[1]}' exceeds capacity at snapshot {idx[0]}: "
                        f"P={float(p_sel.loc[idx]):.6g} MW > p_nom*p_max_pu="
                        f"{float(up_sel.loc[idx]):.6g} MW (excess {worst_high:.6g} MW)."
                    ),
                )
            )
        if worst_low > self.capacity_tol_mw:
            idx = excess_low.idxmax()
            violations.append(
                self.make_violation(
                    name="generator_lower_bound",
                    severity="FATAL",
                    observed=worst_low,
                    threshold=self.capacity_tol_mw,
                    message=(
                        f"Negative generation at '{idx[1]}', snapshot {idx[0]}: "
                        f"P={float(p_sel.loc[idx]):.6g} MW < 0 (violation {worst_low:.6g} MW)."
                    ),
                )
            )

    # ------------------------------------------------------------------ #
    # Check 4: storage energy conservation                               #
    # ------------------------------------------------------------------ #

    def _check_storage_conservation(
        self,
        net: Any,
        violations: List[Any],
        metrics: Dict[str, float],
    ) -> None:
        """Verify discrete SOC recursions for storage units and stores."""
        weights = self._snapshot_weights_hours(net)
        n_units = self._check_storage_units(net, weights, violations, metrics)
        n_stores = self._check_stores(net, weights, violations, metrics)
        metrics["storage_components_checked"] = float(n_units + n_stores)

    def _check_storage_units(
        self,
        net: Any,
        weights: pd.Series,
        violations: List[Any],
        metrics: Dict[str, float],
    ) -> int:
        """Check storage-unit SOC recursion with efficiencies/losses."""
        su = getattr(net, "storage_units", None)
        soc = _component_timeseries(net, "storage_units", "state_of_charge")
        p = _component_timeseries(net, "storage_units", "p")
        if su is None or su.empty or soc is None or p is None:
            return 0
        cols = [c for c in soc.columns if c in su.index]
        worst_resid, worst_info = 0.0, ""

        eff_dispatch = pd.to_numeric(su.get("efficiency_dispatch"), errors="coerce")
        eff_dispatch = eff_dispatch.reindex(cols).where(lambda s: s > 0, 1.0).fillna(1.0)
        eff_store = pd.to_numeric(su.get("efficiency_store"), errors="coerce")
        eff_store = eff_store.reindex(cols).where(lambda s: s > 0, 1.0).fillna(1.0)
        standing = pd.to_numeric(su.get("standing_loss"), errors="coerce")
        standing = standing.reindex(cols).clip(lower=0.0).fillna(0.0)
        inflow = _component_timeseries(net, "storage_units", "inflow")
        spill = _component_timeseries(net, "storage_units", "spill")

        dt = weights.reindex(soc.index).fillna(1.0).astype(float)
        prev_soc = soc[cols].shift(1)
        charge = (-p[cols]).clip(lower=0.0)
        discharge = p[cols].clip(lower=0.0)
        decay = pd.DataFrame(
            1.0 - standing.to_numpy()[None, :] * dt.to_numpy()[:, None],
            index=soc.index,
            columns=cols,
        ).clip(lower=0.0)

        expected = prev_soc * decay
        expected = expected.add(charge.mul(eff_store, axis=1).mul(dt, axis=0), fill_value=0.0)
        expected = expected.sub(discharge.div(eff_dispatch, axis=1).mul(dt, axis=0), fill_value=0.0)
        if inflow is not None:
            expected = expected.add(inflow.reindex(columns=cols, fill_value=0.0).mul(dt, axis=0))
        if spill is not None:
            expected = expected.sub(spill.reindex(columns=cols, fill_value=0.0).mul(dt, axis=0))

        residual = (soc[cols] - expected).iloc[1:]
        scale = max(float(np.nanmax(np.abs(soc[cols].to_numpy()), initial=0.0)), 1e-9)
        if not residual.empty:
            stacked = residual.replace([np.inf, -np.inf], np.nan).stack()
            if not stacked.empty:
                abs_stacked = stacked.abs()
                worst_idx = abs_stacked.idxmax()
                worst_resid = float(abs(stacked.loc[worst_idx]))
                worst_info = f"unit='{worst_idx[1]}' snapshot={worst_idx[0]}"
        metrics["max_storage_soc_residual_mwh"] = worst_resid

        # Initial-state consistency.
        init = pd.to_numeric(su.get("state_of_charge_initial"), errors="coerce").reindex(cols)
        valid_init = init.dropna()
        if not valid_init.empty:
            drift = (soc[cols].iloc[0] - valid_init).abs().max()
            if pd.notna(drift) and float(drift) > self.soc_atol_mwh:
                violations.append(
                    self.make_violation(
                        name="storage_initial_state_mismatch",
                        severity="FATAL",
                        observed=float(drift),
                        threshold=self.soc_atol_mwh,
                        message=(
                            f"First-snapshot SOC deviates from state_of_charge_initial "
                            f"by {float(drift):.6g} MWh."
                        ),
                    )
                )

        if worst_resid > self.soc_atol_mwh + self.soc_rtol * scale:
            violations.append(
                self.make_violation(
                    name="storage_energy_conservation",
                    severity="FATAL",
                    observed=worst_resid,
                    threshold=self.soc_atol_mwh,
                    message=(
                        f"SOC recursion violated for {worst_info}: residual="
                        f"{worst_resid:.6g} MWh after accounting for store/dispatch "
                        f"efficiencies, standing losses, inflow and spillage."
                    ),
                )
            )
        return len(cols)

    def _check_stores(
        self,
        net: Any,
        weights: pd.Series,
        violations: List[Any],
        metrics: Dict[str, float],
    ) -> int:
        """Check store energy recursion: e[t] = e[t-1]*(1-loss) - p*dt."""
        st = getattr(net, "stores", None)
        energy = _component_timeseries(net, "stores", "e")
        p = _component_timeseries(net, "stores", "p")
        if st is None or st.empty or energy is None or p is None:
            return 0
        cols = [c for c in energy.columns if c in st.index]
        dt = weights.reindex(energy.index).fillna(1.0).astype(float)
        standing = pd.to_numeric(st.get("standing_loss"), errors="coerce")
        standing = standing.reindex(cols).clip(lower=0.0).fillna(0.0)
        decay = pd.DataFrame(
            (1.0 - standing.to_numpy()[None, :] * dt.to_numpy()[:, None]),
            index=energy.index,
            columns=cols,
        ).clip(lower=0.0)
        expected = energy[cols].shift(1).mul(decay).sub(p[cols].mul(dt, axis=0), fill_value=0.0)
        residual = (energy[cols] - expected).iloc[1:]
        scale = max(float(np.nanmax(np.abs(energy[cols].to_numpy()), initial=0.0)), 1e-9)
        worst_resid = 0.0
        worst_info = ""
        if not residual.empty:
            stacked = residual.replace([np.inf, -np.inf], np.nan).stack()
            if not stacked.empty:
                abs_stacked = stacked.abs()
                worst_idx = abs_stacked.idxmax()
                worst_resid = float(abs(stacked.loc[worst_idx]))
                worst_info = f"store='{worst_idx[1]}' snapshot={worst_idx[0]}"
        metrics["max_store_energy_residual_mwh"] = worst_resid
        if worst_resid > self.soc_atol_mwh + self.soc_rtol * scale:
            violations.append(
                self.make_violation(
                    name="store_energy_conservation",
                    severity="FATAL",
                    observed=worst_resid,
                    threshold=self.soc_atol_mwh,
                    message=(
                        f"Store energy recursion violated for {worst_info}: residual="
                        f"{worst_resid:.6g} MWh (e[t] != e[t-1]*(1-standing_loss) - p*dt)."
                    ),
                )
            )
        return len(cols)
