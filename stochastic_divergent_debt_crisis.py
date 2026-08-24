from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from keen_model_functions import KeenModel


# =============================================================================
# Scenario 3: stochastic regime selection near the high-debt equilibrium
# =============================================================================
#
# This script replaces the earlier deterministic two-path experiment.
# Every simulated path now uses the full stochastic system with the baseline
# volatility vector (0.05, 0.05, 0.05). All paths begin at the calibrated
# high-debt equilibrium, where the deterministic drift is zero but the frozen
# diffusion matrix is nondegenerate.
#
# The script performs a Monte Carlo ensemble experiment, objectively selects
# a representative crisis realization whose stopping time is closest to the
# median crisis time, and creates one three-panel figure with the same layout
# as the other stress scenarios:
#   (a) state variables over time;
#   (b) three-dimensional phase-space trajectory;
#   (c) financial ratios.
#
# No clipping, projection, reflection, or epsilon floor is imposed.
# =============================================================================


# =============================================================================
# Calibrated model
# =============================================================================
BASELINE_PARAMS: Final[dict[str, float]] = {
    "alpha": 0.02,
    "beta": 0.031,
    "delta": 0.04,
    "nu": 2.7,
    "r": 0.02,
    "eta_p": 0.192,
    "m": 1.875,
    "gamma": 0.9,
    "phi0": -0.292,
    "phi1": 0.469,
    "kappa0": 0.0318,
    "kappa1": 0.575,
    "kappa_min": 0.0,
    "kappa_max": 0.3,
    "dividend0": -0.078,
    "dividend1": 0.553,
    "dividend_min": 0.0,
    "dividend_max": 0.3,
    "sigma_omega": 0.05,
    "sigma_lambda": 0.05,
    "sigma_ell": 0.05,
}

HIGH_DEBT_EQUILIBRIUM: Final[np.ndarray] = np.array(
    [0.3920000000, 0.6543965885, 11.8000000000],
    dtype=float,
)

LOW_DEBT_EQUILIBRIUM: Final[np.ndarray] = np.array(
    [0.6276666667, 0.6724861407, 0.0166666667],
    dtype=float,
)

INITIAL_STATE: Final[np.ndarray] = HIGH_DEBT_EQUILIBRIUM.copy()


# =============================================================================
# Monte Carlo and time-discretisation settings
# =============================================================================
N_PATHS: Final[int] = 1000
MASTER_SEED: Final[int] = 7319
HORIZON: Final[float] = 200.0
DT: Final[float] = 0.01
PLOT_INTERVAL: Final[float] = 0.05

# Crisis and numerical stopping rules
DEBT_CRISIS_THRESHOLD: Final[float] = 1.0e8
COLLAPSE_DEBT_THRESHOLD: Final[float] = 50.0
SHARE_COLLAPSE_THRESHOLD: Final[float] = 1.0e-4
NEGATIVE_DEBT_SAFETY_CUTOFF: Final[float] = -1.0e8

# Figure settings
PHASE_CMAP: Final[str] = "viridis"
FINANCIAL_LINTHRESH: Final[float] = 1.0e-2
OUTPUT_DIRECTORY: Final[Path] = Path("Figures")
DATA_DIRECTORY: Final[Path] = Path("NumericalData")
OUTPUT_STEM: Final[str] = "stochastic_divergent_debt_crisis"

COLORS: Final[dict[str, str]] = {
    "omega": "#1f77b4",          # blue
    "lambda": "#d62728",         # red
    "ell": "#2ca02c",            # green
    "profit": "#1f77b4",         # blue
    "debt_service": "#9467bd",   # purple
    "inflation": "#ff7f0e",      # orange
    "reference": "#666666",      # grey
    "start": "#1f77b4",          # blue
    "end": "#ff7f0e",            # orange
    "low_eq": "#2ca02c",         # green
    "deflation": "#dbeaf4",      # light blue
    "profit_event": "#666666",   # grey
    "investment_event": "#d62728",
}


# =============================================================================
# Outcome codes
# =============================================================================
OUTCOME_ACTIVE: Final[int] = 0
OUTCOME_CRISIS: Final[int] = 1
OUTCOME_UPPER_SHARE_EXIT: Final[int] = 2
OUTCOME_LOWER_SHARE_EXIT: Final[int] = 3
OUTCOME_NEGATIVE_DEBT_EXIT: Final[int] = 4
OUTCOME_NUMERICAL_FAILURE: Final[int] = 5
OUTCOME_HORIZON: Final[int] = 6

OUTCOME_LABELS: Final[dict[int, str]] = {
    OUTCOME_CRISIS: "stochastic crisis",
    OUTCOME_UPPER_SHARE_EXIT: "upper share-space exit",
    OUTCOME_LOWER_SHARE_EXIT: "lower share-space exit",
    OUTCOME_NEGATIVE_DEBT_EXIT: "negative-debt safety cutoff",
    OUTCOME_NUMERICAL_FAILURE: "numerical failure",
    OUTCOME_HORIZON: "completed horizon without terminal event",
}

CRISIS_TRIGGER_NONE: Final[int] = 0
CRISIS_TRIGGER_DEBT: Final[int] = 1
CRISIS_TRIGGER_COLLAPSE: Final[int] = 2

CRISIS_TRIGGER_LABELS: Final[dict[int, str]] = {
    CRISIS_TRIGGER_NONE: "not applicable",
    CRISIS_TRIGGER_DEBT: "debt threshold",
    CRISIS_TRIGGER_COLLAPSE: "share collapse with large positive debt",
}


# =============================================================================
# Data structures
# =============================================================================
@dataclass
class EnsembleResult:
    plot_time: np.ndarray
    history: np.ndarray
    outcome: np.ndarray
    crisis_trigger: np.ndarray
    event_time: np.ndarray
    terminal_states: np.ndarray
    omega_min: np.ndarray
    omega_max: np.ndarray
    lambda_min: np.ndarray
    lambda_max: np.ndarray
    ell_min: np.ndarray
    ell_max: np.ndarray
    profit_min: np.ndarray
    profit_max: np.ndarray
    debt_service_min: np.ndarray
    debt_service_max: np.ndarray
    inflation_min: np.ndarray
    inflation_max: np.ndarray
    first_negative_profit_time: np.ndarray
    first_zero_investment_time: np.ndarray
    first_lambda_below_0_1_time: np.ndarray
    first_ell_above_100_time: np.ndarray
    first_ell_above_10000_time: np.ndarray
    longest_deflation_episode: np.ndarray
    deflation_fraction: np.ndarray
    initial_drift: np.ndarray


# =============================================================================
# Model evaluation
# =============================================================================
def make_model() -> KeenModel:
    """Return the calibrated stochastic Keen model."""
    return KeenModel(BASELINE_PARAMS)


def model_terms(
    model: KeenModel,
    omega: np.ndarray | float,
    lam: np.ndarray | float,
    ell: np.ndarray | float,
) -> dict[str, np.ndarray]:
    """Evaluate the drift and economic diagnostics, allowing arrays."""
    parameters = model.params

    omega_array = np.asarray(omega, dtype=float)
    lambda_array = np.asarray(lam, dtype=float)
    ell_array = np.asarray(ell, dtype=float)

    profit = 1.0 - omega_array - parameters["r"] * ell_array

    investment_raw = (
        parameters["kappa0"]
        + parameters["kappa1"] * profit
    )
    investment = np.clip(
        investment_raw,
        parameters["kappa_min"],
        parameters["kappa_max"],
    )

    dividend_raw = (
        parameters["dividend0"]
        + parameters["dividend1"] * profit
    )
    dividend = np.clip(
        dividend_raw,
        parameters["dividend_min"],
        parameters["dividend_max"],
    )

    inflation = parameters["eta_p"] * (
        parameters["m"] * omega_array - 1.0
    )
    phillips = parameters["phi0"] + parameters["phi1"] * lambda_array

    b_omega = omega_array * (
        phillips
        - parameters["alpha"]
        - (1.0 - parameters["gamma"]) * inflation
    )

    b_lambda = lambda_array * (
        investment / parameters["nu"]
        - parameters["delta"]
        - parameters["alpha"]
        - parameters["beta"]
    )

    interest_growth_gap = (
        parameters["r"]
        - investment / parameters["nu"]
        + parameters["delta"]
        - inflation
    )
    financing_gap = omega_array + investment - 1.0 + dividend
    b_ell = ell_array * interest_growth_gap + financing_gap

    sigma_omega = parameters["sigma_omega"] * np.sqrt(
        np.maximum(omega_array * (1.0 - omega_array), 0.0)
    )
    sigma_lambda = parameters["sigma_lambda"] * np.sqrt(
        np.maximum(lambda_array * (1.0 - lambda_array), 0.0)
    )
    sigma_ell = parameters["sigma_ell"] * np.abs(ell_array)

    return {
        "profit": profit,
        "investment": investment,
        "dividend": dividend,
        "inflation": inflation,
        "phillips": phillips,
        "b_omega": b_omega,
        "b_lambda": b_lambda,
        "b_ell": b_ell,
        "sigma_omega": sigma_omega,
        "sigma_lambda": sigma_lambda,
        "sigma_ell": sigma_ell,
        "interest_growth_gap": interest_growth_gap,
        "financing_gap": financing_gap,
        "debt_service": parameters["r"] * ell_array,
    }


def calculate_path_derived(
    model: KeenModel,
    states: np.ndarray,
) -> dict[str, np.ndarray]:
    """Calculate diagnostics for the representative path."""
    terms = model_terms(
        model,
        states[:, 0],
        states[:, 1],
        states[:, 2],
    )

    ell = states[:, 2]
    proportional_financing = np.full_like(ell, np.nan, dtype=float)
    log_debt_drift = np.full_like(ell, np.nan, dtype=float)

    positive_debt = ell > 0.0
    proportional_financing[positive_debt] = (
        terms["financing_gap"][positive_debt] / ell[positive_debt]
    )
    log_debt_drift[positive_debt] = (
        terms["interest_growth_gap"][positive_debt]
        + proportional_financing[positive_debt]
        - 0.5 * model.params["sigma_ell"] ** 2
    )

    return {
        **terms,
        "proportional_financing": proportional_financing,
        "log_debt_drift": log_debt_drift,
    }


# =============================================================================
# Ensemble simulation
# =============================================================================
def _set_first_time(
    storage: np.ndarray,
    condition: np.ndarray,
    eligible: np.ndarray,
    time_value: float,
) -> None:
    """Set first-event times without overwriting previous events."""
    mask = eligible & condition & np.isnan(storage)
    storage[mask] = time_value


def simulate_ensemble(
    model: KeenModel,
    n_paths: int = N_PATHS,
    horizon: float = HORIZON,
    dt: float = DT,
    plot_interval: float = PLOT_INTERVAL,
    master_seed: int = MASTER_SEED,
    verbose: bool = True,
) -> EnsembleResult:
    """
    Simulate an ensemble of unprojected stochastic paths.

    All paths begin at the high-debt equilibrium. Brownian increments are
    generated for every path at every time step, including paths that have
    already stopped. This fixed draw pattern makes the experiment exactly
    reproducible from the master seed.
    """
    if n_paths <= 0:
        raise ValueError("n_paths must be strictly positive.")
    if horizon <= 0.0 or dt <= 0.0 or plot_interval <= 0.0:
        raise ValueError("horizon, dt, and plot_interval must be positive.")

    step_ratio = horizon / dt
    n_steps = int(round(step_ratio))
    if not np.isclose(step_ratio, n_steps, rtol=0.0, atol=1e-10):
        raise ValueError("horizon / dt must be integral within tolerance.")

    plot_ratio = plot_interval / dt
    plot_stride = int(round(plot_ratio))
    if plot_stride <= 0 or not np.isclose(
        plot_ratio,
        plot_stride,
        rtol=0.0,
        atol=1e-10,
    ):
        raise ValueError("plot_interval / dt must be integral within tolerance.")

    plot_steps = np.arange(0, n_steps + 1, plot_stride, dtype=int)
    if plot_steps[-1] != n_steps:
        plot_steps = np.append(plot_steps, n_steps)
    plot_time = plot_steps.astype(float) * dt

    # Float32 history is sufficient for plotting and keeps memory modest.
    history = np.full(
        (len(plot_steps), n_paths, 3),
        np.nan,
        dtype=np.float32,
    )

    omega = np.full(n_paths, INITIAL_STATE[0], dtype=float)
    lam = np.full(n_paths, INITIAL_STATE[1], dtype=float)
    ell = np.full(n_paths, INITIAL_STATE[2], dtype=float)
    history[0, :, :] = INITIAL_STATE.astype(np.float32)

    active = np.ones(n_paths, dtype=bool)
    outcome = np.full(n_paths, OUTCOME_ACTIVE, dtype=np.int8)
    crisis_trigger = np.full(
        n_paths,
        CRISIS_TRIGGER_NONE,
        dtype=np.int8,
    )
    event_time = np.full(n_paths, np.nan, dtype=float)

    initial_terms = model_terms(model, omega, lam, ell)
    initial_drift = np.array(
        [
            float(np.asarray(initial_terms["b_omega"])[0]),
            float(np.asarray(initial_terms["b_lambda"])[0]),
            float(np.asarray(initial_terms["b_ell"])[0]),
        ],
        dtype=float,
    )

    omega_min = omega.copy()
    omega_max = omega.copy()
    lambda_min = lam.copy()
    lambda_max = lam.copy()
    ell_min = ell.copy()
    ell_max = ell.copy()

    profit_initial = np.asarray(initial_terms["profit"], dtype=float)
    debt_service_initial = np.asarray(
        initial_terms["debt_service"],
        dtype=float,
    )
    inflation_initial = np.asarray(initial_terms["inflation"], dtype=float)

    profit_min = profit_initial.copy()
    profit_max = profit_initial.copy()
    debt_service_min = debt_service_initial.copy()
    debt_service_max = debt_service_initial.copy()
    inflation_min = inflation_initial.copy()
    inflation_max = inflation_initial.copy()

    first_negative_profit_time = np.where(
        profit_initial <= 0.0,
        0.0,
        np.nan,
    )
    first_zero_investment_time = np.where(
        np.asarray(initial_terms["investment"]) <= 1.0e-12,
        0.0,
        np.nan,
    )
    first_lambda_below_0_1_time = np.where(lam <= 0.1, 0.0, np.nan)
    first_ell_above_100_time = np.where(ell >= 100.0, 0.0, np.nan)
    first_ell_above_10000_time = np.where(ell >= 10000.0, 0.0, np.nan)

    current_deflation_steps = np.zeros(n_paths, dtype=np.int32)
    longest_deflation_steps = np.zeros(n_paths, dtype=np.int32)
    total_deflation_steps = np.zeros(n_paths, dtype=np.int32)
    observed_steps = np.zeros(n_paths, dtype=np.int32)

    rng = np.random.default_rng(master_seed)
    sqrt_dt = np.sqrt(dt)
    plot_cursor = 1

    if verbose:
        print("\nSimulating stochastic high-debt ensemble")
        print("=" * 62)
        print(f"Paths: {n_paths}")
        print(f"Horizon: {horizon:.2f} years")
        print(f"Time step: {dt:.4f} years")
        print(f"Master seed: {master_seed}")

    progress_stride = max(1, int(round(20.0 / dt)))

    for step in range(1, n_steps + 1):
        time_value = step * dt
        active_before = active.copy()

        # Fixed-size draws preserve exact reproducibility even after paths stop.
        dW = rng.standard_normal((n_paths, 3)) * sqrt_dt

        active_indices = np.flatnonzero(active_before)
        if active_indices.size > 0:
            omega_active = omega[active_indices]
            lambda_active = lam[active_indices]
            ell_active = ell[active_indices]

            terms = model_terms(
                model,
                omega_active,
                lambda_active,
                ell_active,
            )

            omega_new = (
                omega_active
                + terms["b_omega"] * dt
                + terms["sigma_omega"] * dW[active_indices, 0]
            )
            lambda_new = (
                lambda_active
                + terms["b_lambda"] * dt
                + terms["sigma_lambda"] * dW[active_indices, 1]
            )
            ell_new = (
                ell_active
                + terms["b_ell"] * dt
                + terms["sigma_ell"] * dW[active_indices, 2]
            )

            omega[active_indices] = omega_new
            lam[active_indices] = lambda_new
            ell[active_indices] = ell_new

            finite_local = (
                np.isfinite(omega_new)
                & np.isfinite(lambda_new)
                & np.isfinite(ell_new)
            )
            finite_indices = active_indices[finite_local]
            failure_indices = active_indices[~finite_local]

            if failure_indices.size > 0:
                outcome[failure_indices] = OUTCOME_NUMERICAL_FAILURE
                event_time[failure_indices] = time_value
                active[failure_indices] = False

            if finite_indices.size > 0:
                finite_terms = model_terms(
                    model,
                    omega[finite_indices],
                    lam[finite_indices],
                    ell[finite_indices],
                )

                # State and financial ranges
                omega_min[finite_indices] = np.minimum(
                    omega_min[finite_indices],
                    omega[finite_indices],
                )
                omega_max[finite_indices] = np.maximum(
                    omega_max[finite_indices],
                    omega[finite_indices],
                )
                lambda_min[finite_indices] = np.minimum(
                    lambda_min[finite_indices],
                    lam[finite_indices],
                )
                lambda_max[finite_indices] = np.maximum(
                    lambda_max[finite_indices],
                    lam[finite_indices],
                )
                ell_min[finite_indices] = np.minimum(
                    ell_min[finite_indices],
                    ell[finite_indices],
                )
                ell_max[finite_indices] = np.maximum(
                    ell_max[finite_indices],
                    ell[finite_indices],
                )

                profit_values = np.asarray(
                    finite_terms["profit"],
                    dtype=float,
                )
                debt_service_values = np.asarray(
                    finite_terms["debt_service"],
                    dtype=float,
                )
                inflation_values = np.asarray(
                    finite_terms["inflation"],
                    dtype=float,
                )
                investment_values = np.asarray(
                    finite_terms["investment"],
                    dtype=float,
                )

                profit_min[finite_indices] = np.minimum(
                    profit_min[finite_indices],
                    profit_values,
                )
                profit_max[finite_indices] = np.maximum(
                    profit_max[finite_indices],
                    profit_values,
                )
                debt_service_min[finite_indices] = np.minimum(
                    debt_service_min[finite_indices],
                    debt_service_values,
                )
                debt_service_max[finite_indices] = np.maximum(
                    debt_service_max[finite_indices],
                    debt_service_values,
                )
                inflation_min[finite_indices] = np.minimum(
                    inflation_min[finite_indices],
                    inflation_values,
                )
                inflation_max[finite_indices] = np.maximum(
                    inflation_max[finite_indices],
                    inflation_values,
                )

                finite_global_mask = np.zeros(n_paths, dtype=bool)
                finite_global_mask[finite_indices] = True

                condition = np.zeros(n_paths, dtype=bool)
                condition[finite_indices] = profit_values <= 0.0
                _set_first_time(
                    first_negative_profit_time,
                    condition,
                    finite_global_mask,
                    time_value,
                )

                condition[:] = False
                condition[finite_indices] = investment_values <= 1.0e-12
                _set_first_time(
                    first_zero_investment_time,
                    condition,
                    finite_global_mask,
                    time_value,
                )

                condition[:] = False
                condition[finite_indices] = lam[finite_indices] <= 0.1
                _set_first_time(
                    first_lambda_below_0_1_time,
                    condition,
                    finite_global_mask,
                    time_value,
                )

                condition[:] = False
                condition[finite_indices] = ell[finite_indices] >= 100.0
                _set_first_time(
                    first_ell_above_100_time,
                    condition,
                    finite_global_mask,
                    time_value,
                )

                condition[:] = False
                condition[finite_indices] = ell[finite_indices] >= 10000.0
                _set_first_time(
                    first_ell_above_10000_time,
                    condition,
                    finite_global_mask,
                    time_value,
                )

                # Deflation diagnostics
                deflation = inflation_values < 0.0
                observed_steps[finite_indices] += 1
                total_deflation_steps[finite_indices] += deflation.astype(
                    np.int32
                )
                current_deflation_steps[finite_indices] = np.where(
                    deflation,
                    current_deflation_steps[finite_indices] + 1,
                    0,
                )
                longest_deflation_steps[finite_indices] = np.maximum(
                    longest_deflation_steps[finite_indices],
                    current_deflation_steps[finite_indices],
                )

                # Outcome classification. Crisis is checked before lower exits
                # so that a near-zero share with large debt is recorded as the
                # stochastic Keen crisis rather than as a generic boundary exit.
                debt_trigger = ell[finite_indices] >= DEBT_CRISIS_THRESHOLD
                collapse_trigger = (
                    (
                        (omega[finite_indices] <= SHARE_COLLAPSE_THRESHOLD)
                        | (lam[finite_indices] <= SHARE_COLLAPSE_THRESHOLD)
                    )
                    & (ell[finite_indices] >= COLLAPSE_DEBT_THRESHOLD)
                )
                crisis_local = debt_trigger | collapse_trigger

                upper_exit_local = (
                    ~crisis_local
                    & (
                        (omega[finite_indices] >= 1.0)
                        | (lam[finite_indices] >= 1.0)
                    )
                )
                lower_exit_local = (
                    ~crisis_local
                    & ~upper_exit_local
                    & (
                        (omega[finite_indices] <= 0.0)
                        | (lam[finite_indices] <= 0.0)
                    )
                )
                negative_debt_local = (
                    ~crisis_local
                    & ~upper_exit_local
                    & ~lower_exit_local
                    & (ell[finite_indices] <= NEGATIVE_DEBT_SAFETY_CUTOFF)
                )

                crisis_indices = finite_indices[crisis_local]
                upper_indices = finite_indices[upper_exit_local]
                lower_indices = finite_indices[lower_exit_local]
                negative_indices = finite_indices[negative_debt_local]

                if crisis_indices.size > 0:
                    outcome[crisis_indices] = OUTCOME_CRISIS
                    event_time[crisis_indices] = time_value
                    active[crisis_indices] = False

                    debt_members = finite_indices[debt_trigger]
                    collapse_members = finite_indices[
                        collapse_trigger & ~debt_trigger
                    ]
                    crisis_trigger[debt_members] = CRISIS_TRIGGER_DEBT
                    crisis_trigger[
                        collapse_members
                    ] = CRISIS_TRIGGER_COLLAPSE

                if upper_indices.size > 0:
                    outcome[upper_indices] = OUTCOME_UPPER_SHARE_EXIT
                    event_time[upper_indices] = time_value
                    active[upper_indices] = False

                if lower_indices.size > 0:
                    outcome[lower_indices] = OUTCOME_LOWER_SHARE_EXIT
                    event_time[lower_indices] = time_value
                    active[lower_indices] = False

                if negative_indices.size > 0:
                    outcome[
                        negative_indices
                    ] = OUTCOME_NEGATIVE_DEBT_EXIT
                    event_time[negative_indices] = time_value
                    active[negative_indices] = False

        if plot_cursor < len(plot_steps) and step == plot_steps[plot_cursor]:
            history[plot_cursor, :, 0] = omega.astype(np.float32)
            history[plot_cursor, :, 1] = lam.astype(np.float32)
            history[plot_cursor, :, 2] = ell.astype(np.float32)
            plot_cursor += 1

        if verbose and (step % progress_stride == 0 or step == n_steps):
            print(
                f"  t={time_value:6.1f}: "
                f"active={int(active.sum()):4d}, "
                f"crisis={int(np.sum(outcome == OUTCOME_CRISIS)):4d}"
            )

        if not np.any(active):
            # Fill any remaining plotting rows with terminal states.
            while plot_cursor < len(plot_steps):
                history[plot_cursor, :, 0] = omega.astype(np.float32)
                history[plot_cursor, :, 1] = lam.astype(np.float32)
                history[plot_cursor, :, 2] = ell.astype(np.float32)
                plot_cursor += 1
            break

    horizon_indices = np.flatnonzero(active)
    if horizon_indices.size > 0:
        outcome[horizon_indices] = OUTCOME_HORIZON
        event_time[horizon_indices] = horizon
        active[horizon_indices] = False

    terminal_states = np.column_stack([omega, lam, ell])

    with np.errstate(divide="ignore", invalid="ignore"):
        deflation_fraction = np.divide(
            total_deflation_steps,
            observed_steps,
            out=np.zeros(n_paths, dtype=float),
            where=observed_steps > 0,
        )

    return EnsembleResult(
        plot_time=plot_time,
        history=history,
        outcome=outcome,
        crisis_trigger=crisis_trigger,
        event_time=event_time,
        terminal_states=terminal_states,
        omega_min=omega_min,
        omega_max=omega_max,
        lambda_min=lambda_min,
        lambda_max=lambda_max,
        ell_min=ell_min,
        ell_max=ell_max,
        profit_min=profit_min,
        profit_max=profit_max,
        debt_service_min=debt_service_min,
        debt_service_max=debt_service_max,
        inflation_min=inflation_min,
        inflation_max=inflation_max,
        first_negative_profit_time=first_negative_profit_time,
        first_zero_investment_time=first_zero_investment_time,
        first_lambda_below_0_1_time=first_lambda_below_0_1_time,
        first_ell_above_100_time=first_ell_above_100_time,
        first_ell_above_10000_time=first_ell_above_10000_time,
        longest_deflation_episode=longest_deflation_steps.astype(float) * dt,
        deflation_fraction=deflation_fraction,
        initial_drift=initial_drift,
    )


# =============================================================================
# Representative crisis path
# =============================================================================
def select_representative_crisis(result: EnsembleResult) -> int:
    """Select the positive-debt crisis path nearest the median crisis time."""
    crisis_indices = np.flatnonzero(result.outcome == OUTCOME_CRISIS)
    if crisis_indices.size == 0:
        raise RuntimeError(
            "No stochastic crisis path was observed. Increase N_PATHS or "
            "HORIZON, or inspect the outcome classification."
        )

    # Prefer paths whose debt remains positive so the logarithmic debt axis is
    # well defined throughout the displayed realization.
    positive_debt_candidates = crisis_indices[
        result.ell_min[crisis_indices] > 0.0
    ]
    candidates = (
        positive_debt_candidates
        if positive_debt_candidates.size > 0
        else crisis_indices
    )

    crisis_times = result.event_time[candidates]
    median_time = float(np.median(crisis_times))
    selected_position = int(np.argmin(np.abs(crisis_times - median_time)))
    return int(candidates[selected_position])


def extract_path(
    result: EnsembleResult,
    path_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract the stored path and append its exact terminal event state."""
    stop_time = float(result.event_time[path_index])
    mask = result.plot_time <= stop_time + 1.0e-12
    time = result.plot_time[mask].astype(float)
    states = result.history[mask, path_index, :].astype(float)

    finite_rows = np.all(np.isfinite(states), axis=1)
    time = time[finite_rows]
    states = states[finite_rows]

    terminal_state = result.terminal_states[path_index].astype(float)
    if time.size == 0 or not np.isclose(
        time[-1],
        stop_time,
        rtol=0.0,
        atol=1.0e-12,
    ):
        time = np.append(time, stop_time)
        states = np.vstack([states, terminal_state])
    else:
        states[-1] = terminal_state

    return time, states


# =============================================================================
# Summary statistics
# =============================================================================
def optional_float(value: float) -> float | None:
    """Convert NaN to None for JSON and readable printing."""
    return None if np.isnan(value) else float(value)


def build_summary(
    model: KeenModel,
    result: EnsembleResult,
    selected_index: int,
) -> dict[str, object]:
    """Build ensemble and representative-path diagnostics."""
    counts = {
        label: int(np.sum(result.outcome == code))
        for code, label in OUTCOME_LABELS.items()
    }
    percentages = {
        label: 100.0 * count / len(result.outcome)
        for label, count in counts.items()
    }

    crisis_indices = np.flatnonzero(result.outcome == OUTCOME_CRISIS)
    crisis_times = result.event_time[crisis_indices]

    terminal_state = result.terminal_states[selected_index]
    terminal_terms = model_terms(
        model,
        terminal_state[0],
        terminal_state[1],
        terminal_state[2],
    )

    terminal_ell = float(terminal_state[2])
    terminal_proportional_financing = (
        float(terminal_terms["financing_gap"]) / terminal_ell
        if terminal_ell > 0.0
        else None
    )
    terminal_log_debt_drift = (
        float(terminal_terms["interest_growth_gap"])
        + terminal_proportional_financing
        - 0.5 * model.params["sigma_ell"] ** 2
        if terminal_proportional_financing is not None
        else None
    )

    trigger_code = int(result.crisis_trigger[selected_index])

    return {
        "n_paths": int(len(result.outcome)),
        "master_seed": MASTER_SEED,
        "horizon": HORIZON,
        "dt": DT,
        "plot_interval": PLOT_INTERVAL,
        "initial_state": INITIAL_STATE.tolist(),
        "initial_drift": result.initial_drift.tolist(),
        "initial_drift_norm": float(np.linalg.norm(result.initial_drift)),
        "volatility_vector": [
            model.params["sigma_omega"],
            model.params["sigma_lambda"],
            model.params["sigma_ell"],
        ],
        "debt_crisis_threshold": DEBT_CRISIS_THRESHOLD,
        "collapse_debt_threshold": COLLAPSE_DEBT_THRESHOLD,
        "share_collapse_threshold": SHARE_COLLAPSE_THRESHOLD,
        "outcome_counts": counts,
        "outcome_percentages": percentages,
        "crisis_time_min": float(np.min(crisis_times)),
        "crisis_time_q25": float(np.quantile(crisis_times, 0.25)),
        "crisis_time_median": float(np.median(crisis_times)),
        "crisis_time_q75": float(np.quantile(crisis_times, 0.75)),
        "crisis_time_max": float(np.max(crisis_times)),
        "selected_path_index_zero_based": int(selected_index),
        "selected_path_index_one_based": int(selected_index + 1),
        "selected_crisis_trigger": CRISIS_TRIGGER_LABELS[trigger_code],
        "selected_crisis_time": float(result.event_time[selected_index]),
        "selected_terminal_state": terminal_state.tolist(),
        "selected_omega_range": [
            float(result.omega_min[selected_index]),
            float(result.omega_max[selected_index]),
        ],
        "selected_lambda_range": [
            float(result.lambda_min[selected_index]),
            float(result.lambda_max[selected_index]),
        ],
        "selected_ell_range": [
            float(result.ell_min[selected_index]),
            float(result.ell_max[selected_index]),
        ],
        "selected_profit_range": [
            float(result.profit_min[selected_index]),
            float(result.profit_max[selected_index]),
        ],
        "selected_debt_service_range": [
            float(result.debt_service_min[selected_index]),
            float(result.debt_service_max[selected_index]),
        ],
        "selected_inflation_range": [
            float(result.inflation_min[selected_index]),
            float(result.inflation_max[selected_index]),
        ],
        "selected_first_negative_profit_time": optional_float(
            result.first_negative_profit_time[selected_index]
        ),
        "selected_first_zero_investment_time": optional_float(
            result.first_zero_investment_time[selected_index]
        ),
        "selected_first_lambda_below_0_1_time": optional_float(
            result.first_lambda_below_0_1_time[selected_index]
        ),
        "selected_first_ell_above_100_time": optional_float(
            result.first_ell_above_100_time[selected_index]
        ),
        "selected_first_ell_above_10000_time": optional_float(
            result.first_ell_above_10000_time[selected_index]
        ),
        "selected_longest_deflation_episode": float(
            result.longest_deflation_episode[selected_index]
        ),
        "selected_deflation_fraction": float(
            result.deflation_fraction[selected_index]
        ),
        "terminal_interest_growth_gap": float(
            terminal_terms["interest_growth_gap"]
        ),
        "terminal_financing_gap": float(terminal_terms["financing_gap"]),
        "terminal_proportional_financing": terminal_proportional_financing,
        "ito_log_debt_correction": -0.5 * model.params["sigma_ell"] ** 2,
        "terminal_log_debt_drift": terminal_log_debt_drift,
    }


# =============================================================================
# Data export
# =============================================================================
def save_outputs(
    result: EnsembleResult,
    selected_index: int,
    selected_time: np.ndarray,
    selected_states: np.ndarray,
    selected_derived: dict[str, np.ndarray],
    summary: dict[str, object],
) -> tuple[Path, Path, Path]:
    """Save ensemble outcomes, representative path, and summary JSON."""
    DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)

    ensemble_path = DATA_DIRECTORY / f"{OUTPUT_STEM}_ensemble.csv"
    selected_path = DATA_DIRECTORY / f"{OUTPUT_STEM}_representative_path.csv"
    summary_path = DATA_DIRECTORY / f"{OUTPUT_STEM}_statistics.json"

    ensemble_table = np.column_stack(
        [
            np.arange(len(result.outcome), dtype=int),
            result.outcome,
            result.crisis_trigger,
            result.event_time,
            result.terminal_states,
            result.omega_min,
            result.omega_max,
            result.lambda_min,
            result.lambda_max,
            result.ell_min,
            result.ell_max,
            result.profit_min,
            result.profit_max,
            result.debt_service_min,
            result.debt_service_max,
            result.inflation_min,
            result.inflation_max,
            result.first_negative_profit_time,
            result.first_zero_investment_time,
            result.first_lambda_below_0_1_time,
            result.first_ell_above_100_time,
            result.first_ell_above_10000_time,
            result.longest_deflation_episode,
            result.deflation_fraction,
        ]
    )
    ensemble_header = (
        "path_index_zero_based,outcome_code,crisis_trigger_code,event_time,"
        "terminal_omega,terminal_lambda,terminal_ell,omega_min,omega_max,"
        "lambda_min,lambda_max,ell_min,ell_max,profit_min,profit_max,"
        "debt_service_min,debt_service_max,inflation_min,inflation_max,"
        "first_negative_profit_time,first_zero_investment_time,"
        "first_lambda_below_0_1_time,first_ell_above_100_time,"
        "first_ell_above_10000_time,longest_deflation_episode,"
        "deflation_fraction"
    )
    np.savetxt(
        ensemble_path,
        ensemble_table,
        delimiter=",",
        header=ensemble_header,
        comments="",
    )

    representative_table = np.column_stack(
        [
            selected_time,
            selected_states,
            selected_derived["profit"],
            selected_derived["investment"],
            selected_derived["dividend"],
            selected_derived["inflation"],
            selected_derived["debt_service"],
            selected_derived["interest_growth_gap"],
            selected_derived["financing_gap"],
            selected_derived["proportional_financing"],
            selected_derived["log_debt_drift"],
        ]
    )
    representative_header = (
        "time,omega,lambda,ell,profit_share,investment,dividend,inflation,"
        "debt_service,interest_growth_gap,financing_gap,"
        "proportional_financing,ito_log_debt_drift"
    )
    np.savetxt(
        selected_path,
        representative_table,
        delimiter=",",
        header=representative_header,
        comments="",
    )

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    return ensemble_path, selected_path, summary_path


# =============================================================================
# Plot utilities
# =============================================================================
def coloured_3d_phase_line(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    time: np.ndarray,
    linewidth: float = 2.3,
) -> Line3DCollection:
    """Add a time-coloured three-dimensional phase trajectory."""
    points = np.column_stack([x, y, z]).reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    collection = Line3DCollection(
        segments,
        cmap=PHASE_CMAP,
        norm=Normalize(float(time[0]), float(time[-1])),
        linewidth=linewidth,
        zorder=3,
    )
    collection.set_array(time[:-1])
    ax.add_collection3d(collection)
    ax.autoscale_view()
    return collection


def padded_bounds(
    values: np.ndarray,
    fraction: float = 0.05,
) -> tuple[float, float]:
    """Return data-driven bounds with modest padding."""
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    span = maximum - minimum
    padding = fraction * span if span > 0.0 else 0.05
    return minimum - padding, maximum + padding


def true_intervals(
    time: np.ndarray,
    mask: np.ndarray,
) -> list[tuple[float, float]]:
    """Return contiguous time intervals over which mask is true."""
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0 or not np.any(mask):
        return []

    changes = np.diff(mask.astype(np.int8))
    starts = list(np.flatnonzero(changes == 1) + 1)
    ends = list(np.flatnonzero(changes == -1) + 1)

    if mask[0]:
        starts.insert(0, 0)
    if mask[-1]:
        ends.append(len(mask))

    intervals: list[tuple[float, float]] = []
    for start, end in zip(starts, ends):
        start_time = float(time[start])
        end_time = float(time[min(end, len(time) - 1)])
        intervals.append((start_time, end_time))
    return intervals


def add_event_line(
    ax: plt.Axes,
    time_value: float | None,
    color: str,
    linestyle: str,
    label: str,
) -> Line2D | None:
    """Add a labelled event line when the event was observed."""
    if time_value is None:
        return None
    return ax.axvline(
        time_value,
        color=color,
        linestyle=linestyle,
        linewidth=1.4,
        alpha=0.9,
        label=label,
    )


# =============================================================================
# Main three-panel figure
# =============================================================================
def create_main_figure(
    selected_time: np.ndarray,
    selected_states: np.ndarray,
    selected_derived: dict[str, np.ndarray],
    summary: dict[str, object],
) -> tuple[plt.Figure, Path, Path]:
    """Create the thesis figure using the same three-panel structure."""
    omega = selected_states[:, 0]
    lam = selected_states[:, 1]
    ell = selected_states[:, 2]

    if np.any(ell <= 0.0):
        raise ValueError(
            "The representative crisis path contains non-positive debt, "
            "so the logarithmic debt display is not valid."
        )

    log_ell = np.log10(ell)
    high_log_ell = float(np.log10(HIGH_DEBT_EQUILIBRIUM[2]))
    low_log_ell = float(np.log10(LOW_DEBT_EQUILIBRIUM[2]))

    figure = plt.figure(figsize=(21.5, 7.6))
    grid = figure.add_gridspec(
        1,
        3,
        width_ratios=[1.10, 1.20, 1.15],
        left=0.045,
        right=0.985,
        top=0.84,
        bottom=0.12,
        wspace=0.38,
    )

    ax1 = figure.add_subplot(grid[0, 0])
    ax2 = figure.add_subplot(grid[0, 1], projection="3d")
    ax3 = figure.add_subplot(grid[0, 2])

    # ------------------------------------------------------------------
    # Panel (a): state variables
    # ------------------------------------------------------------------
    omega_line = ax1.plot(
        selected_time,
        omega,
        color=COLORS["omega"],
        linewidth=2.0,
        label=r"Wage share $(\omega_t)$",
        zorder=3,
    )[0]
    lambda_line = ax1.plot(
        selected_time,
        lam,
        color=COLORS["lambda"],
        linewidth=2.0,
        label=r"Employment $(\lambda_t)$",
        zorder=3,
    )[0]

    ax1.axhline(
        0.0,
        color=COLORS["reference"],
        linestyle="--",
        linewidth=1.0,
        alpha=0.7,
    )
    ax1.axhline(
        1.0,
        color=COLORS["reference"],
        linestyle="--",
        linewidth=1.0,
        alpha=0.7,
    )
    ax1.set_xlim(0.0, float(selected_time[-1]))
    ax1.set_xlabel("Time (years)", fontsize=14)
    ax1.set_ylabel(
        r"Share variables $(\omega_t,\lambda_t)$",
        fontsize=14,
    )
    ax1.set_title("(a) State Variables", fontsize=17, fontweight="bold")
    ax1.grid(True, alpha=0.25)

    debt_axis = ax1.twinx()
    debt_line = debt_axis.plot(
        selected_time,
        ell,
        color=COLORS["ell"],
        linewidth=2.2,
        label=r"Net debt $(\ell_t)$",
        zorder=3,
    )[0]
    debt_axis.set_yscale("log")
    debt_axis.set_ylabel(
        r"Net debt $(\ell_t)$, logarithmic scale",
        fontsize=14,
        color=COLORS["ell"],
    )
    debt_axis.tick_params(axis="y", colors=COLORS["ell"])
    debt_axis.spines["right"].set_color(COLORS["ell"])

    ax1.legend(
        [omega_line, lambda_line, debt_line],
        [
            omega_line.get_label(),
            lambda_line.get_label(),
            debt_line.get_label(),
        ],
        loc="best",
        fontsize=10.5,
        framealpha=0.95,
    )

    # ------------------------------------------------------------------
    # Panel (b): three-dimensional phase-space trajectory
    # ------------------------------------------------------------------
    phase_collection = coloured_3d_phase_line(
        ax2,
        omega,
        lam,
        log_ell,
        selected_time,
    )

    ax2.scatter(
        [omega[0]],
        [lam[0]],
        [log_ell[0]],
        s=90,
        marker="o",
        color=COLORS["start"],
        edgecolor="none",
        zorder=6,
    )
    ax2.scatter(
        [omega[-1]],
        [lam[-1]],
        [log_ell[-1]],
        s=95,
        marker="X",
        color=COLORS["end"],
        edgecolor="none",
        zorder=6,
    )
    ax2.scatter(
        [LOW_DEBT_EQUILIBRIUM[0]],
        [LOW_DEBT_EQUILIBRIUM[1]],
        [low_log_ell],
        s=135,
        marker="*",
        color=COLORS["low_eq"],
        edgecolor="none",
        zorder=6,
    )

    all_omega = np.concatenate([omega, [LOW_DEBT_EQUILIBRIUM[0]]])
    all_lambda = np.concatenate([lam, [LOW_DEBT_EQUILIBRIUM[1]]])
    all_log_ell = np.concatenate([log_ell, [high_log_ell, low_log_ell]])

    ax2.set_xlim(*padded_bounds(all_omega, 0.04))
    ax2.set_ylim(*padded_bounds(all_lambda, 0.04))
    ax2.set_zlim(*padded_bounds(all_log_ell, 0.04))

    exponent_min = int(np.floor(np.min(all_log_ell)))
    exponent_max = int(np.ceil(np.max(all_log_ell)))
    exponent_step = max(1, int(np.ceil((exponent_max - exponent_min) / 6)))
    exponents = np.arange(
        exponent_min,
        exponent_max + 1,
        exponent_step,
        dtype=int,
    )
    ax2.set_zticks(exponents)
    ax2.set_zticklabels([rf"$10^{{{value}}}$" for value in exponents])

    ax2.set_xlabel(r"Wage share $(\omega_t)$", fontsize=12, labelpad=8)
    ax2.set_ylabel(r"Employment $(\lambda_t)$", fontsize=12, labelpad=8)
    ax2.set_zlabel(
        r"Net debt $(\ell_t)$ (log scale)",
        fontsize=11,
        labelpad=3,
    )
    ax2.set_title(
        "(b) Phase-Space Trajectory",
        fontsize=17,
        fontweight="bold",
    )
    ax2.view_init(elev=24, azim=-58)
    ax2.grid(True, alpha=0.25)

    ax2.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=COLORS["start"],
                markersize=8,
                label=r"Start at $x_H^*$",
            ),
            Line2D(
                [0],
                [0],
                marker="X",
                color="none",
                markerfacecolor=COLORS["end"],
                markersize=8,
                label="Crisis stopping point",
            ),
            Line2D(
                [0],
                [0],
                marker="*",
                color="none",
                markerfacecolor=COLORS["low_eq"],
                markersize=11,
                label=r"Low-debt equilibrium $x_L^*$",
            ),
        ],
        loc="upper left",
        fontsize=9.2,
        framealpha=0.95,
    )

    colour_bar = figure.colorbar(
        phase_collection,
        ax=ax2,
        fraction=0.047,
        pad=0.08,
    )
    colour_bar.set_label("Time (years)", fontsize=11)
    colour_bar.ax.tick_params(labelsize=9)

    # ------------------------------------------------------------------
    # Panel (c): financial ratios
    # ------------------------------------------------------------------
    for start, end in true_intervals(
        selected_time,
        selected_derived["inflation"] < 0.0,
    ):
        ax3.axvspan(
            start,
            end,
            color=COLORS["deflation"],
            alpha=0.45,
            linewidth=0.0,
            zorder=0,
        )

    debt_service_line = ax3.plot(
        selected_time,
        selected_derived["debt_service"],
        color=COLORS["debt_service"],
        linewidth=1.9,
        label=r"Debt service $(r\ell_t)$",
        zorder=3,
    )[0]
    profit_line = ax3.plot(
        selected_time,
        selected_derived["profit"],
        color=COLORS["profit"],
        linewidth=1.9,
        label=r"Profit share $(\pi_t)$",
        zorder=3,
    )[0]
    inflation_line = ax3.plot(
        selected_time,
        selected_derived["inflation"],
        color=COLORS["inflation"],
        linewidth=1.8,
        label=r"Inflation $(i(\omega_t))$",
        zorder=3,
    )[0]

    ax3.set_yscale(
        "symlog",
        linthresh=FINANCIAL_LINTHRESH,
        linscale=1.0,
        base=10,
    )
    ax3.axhline(
        0.0,
        color=COLORS["reference"],
        linestyle="--",
        linewidth=1.0,
        alpha=0.7,
    )

    event_handles: list[Line2D] = []
    negative_profit_time = summary["selected_first_negative_profit_time"]
    zero_investment_time = summary["selected_first_zero_investment_time"]

    profit_event = add_event_line(
        ax3,
        negative_profit_time,
        COLORS["profit_event"],
        ":",
        (
            rf"$\pi_t\leq0$ at $t\approx{negative_profit_time:.2f}$"
            if negative_profit_time is not None
            else ""
        ),
    )
    if profit_event is not None:
        event_handles.append(profit_event)

    investment_event = add_event_line(
        ax3,
        zero_investment_time,
        COLORS["investment_event"],
        "-.",
        (
            rf"$\kappa(\pi_t)=0$ at $t\approx{zero_investment_time:.2f}$"
            if zero_investment_time is not None
            else ""
        ),
    )
    if investment_event is not None:
        event_handles.append(investment_event)

    deflation_proxy = Line2D(
        [0],
        [0],
        color=COLORS["deflation"],
        linewidth=8,
        alpha=0.6,
        label="Deflation episode",
    )

    ax3.set_xlim(0.0, float(selected_time[-1]))
    ax3.set_xlabel("Time (years)", fontsize=14)
    ax3.set_ylabel("Ratio (symmetric logarithmic scale)", fontsize=14)
    ax3.set_title("(c) Financial Ratios", fontsize=17, fontweight="bold")
    ax3.grid(True, alpha=0.25)

    ax3.legend(
        [
            debt_service_line,
            profit_line,
            inflation_line,
            deflation_proxy,
            *event_handles,
        ],
        [
            debt_service_line.get_label(),
            profit_line.get_label(),
            inflation_line.get_label(),
            deflation_proxy.get_label(),
            *[handle.get_label() for handle in event_handles],
        ],
        loc="best",
        fontsize=9.5,
        framealpha=0.95,
    )

    figure.suptitle(
        "Stochastic Divergent-Debt Crisis near the High-Debt Equilibrium\n"
        rf"Baseline volatilities; master seed {MASTER_SEED}; "
        rf"representative ensemble member "
        rf"{summary['selected_path_index_one_based']}",
        fontsize=20,
        fontweight="bold",
        y=0.97,
    )

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    png_path = OUTPUT_DIRECTORY / f"{OUTPUT_STEM}.png"
    pdf_path = OUTPUT_DIRECTORY / f"{OUTPUT_STEM}.pdf"
    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")

    return figure, png_path, pdf_path


# =============================================================================
# Printed diagnostics
# =============================================================================
def format_optional_time(value: float | None) -> str:
    """Format an optional event time."""
    return "not observed" if value is None else f"{value:.2f} years"


def print_summary(summary: dict[str, object]) -> None:
    """Print the diagnostics needed to rewrite Scenario 3."""
    print("\nStochastic regime selection near the high-debt equilibrium")
    print("=" * 72)
    print(f"Paths: {summary['n_paths']}")
    print(f"Master seed: {summary['master_seed']}")
    print(f"Horizon: {summary['horizon']:.2f} years")
    print(f"Time step: {summary['dt']:.4f} years")
    print(
        "Initial state: "
        f"{tuple(round(x, 8) for x in summary['initial_state'])}"
    )
    print(
        "Volatility vector: "
        f"{tuple(summary['volatility_vector'])}"
    )
    print(
        "Initial deterministic drift norm: "
        f"{summary['initial_drift_norm']:.10e}"
    )

    print("\nEnsemble outcomes")
    print("-" * 72)
    counts = summary["outcome_counts"]
    percentages = summary["outcome_percentages"]
    for label in OUTCOME_LABELS.values():
        print(
            f"{label}: {counts[label]:4d} "
            f"({percentages[label]:6.2f}%)"
        )

    print("\nCrisis-time distribution")
    print("-" * 72)
    print(f"Minimum: {summary['crisis_time_min']:.2f} years")
    print(f"25th percentile: {summary['crisis_time_q25']:.2f} years")
    print(f"Median: {summary['crisis_time_median']:.2f} years")
    print(f"75th percentile: {summary['crisis_time_q75']:.2f} years")
    print(f"Maximum: {summary['crisis_time_max']:.2f} years")

    print("\nRepresentative crisis path")
    print("-" * 72)
    print(
        "Ensemble member: "
        f"{summary['selected_path_index_one_based']}"
    )
    print(f"Trigger: {summary['selected_crisis_trigger']}")
    print(f"Crisis time: {summary['selected_crisis_time']:.4f} years")
    print(
        "Terminal state: "
        f"{tuple(summary['selected_terminal_state'])}"
    )
    print(
        "omega range:  "
        f"[{summary['selected_omega_range'][0]:.8f}, "
        f"{summary['selected_omega_range'][1]:.8f}]"
    )
    print(
        "lambda range: "
        f"[{summary['selected_lambda_range'][0]:.8f}, "
        f"{summary['selected_lambda_range'][1]:.8f}]"
    )
    print(
        "ell range:    "
        f"[{summary['selected_ell_range'][0]:.8e}, "
        f"{summary['selected_ell_range'][1]:.8e}]"
    )

    print("\nRepresentative financial ranges")
    print("-" * 72)
    print(
        "Profit share: "
        f"[{summary['selected_profit_range'][0]:.8f}, "
        f"{summary['selected_profit_range'][1]:.8f}]"
    )
    print(
        "Debt service: "
        f"[{summary['selected_debt_service_range'][0]:.8e}, "
        f"{summary['selected_debt_service_range'][1]:.8e}]"
    )
    print(
        "Inflation:    "
        f"[{summary['selected_inflation_range'][0]:.8f}, "
        f"{summary['selected_inflation_range'][1]:.8f}]"
    )
    print(
        "Deflation share of observed path: "
        f"{100.0 * summary['selected_deflation_fraction']:.2f}%"
    )
    print(
        "Longest deflation episode: "
        f"{summary['selected_longest_deflation_episode']:.2f} years"
    )

    print("\nRepresentative transition times")
    print("-" * 72)
    print(
        "Profit share becomes non-positive: "
        f"{format_optional_time(summary['selected_first_negative_profit_time'])}"
    )
    print(
        "Investment reaches zero: "
        f"{format_optional_time(summary['selected_first_zero_investment_time'])}"
    )
    print(
        "Employment falls below 0.1: "
        f"{format_optional_time(summary['selected_first_lambda_below_0_1_time'])}"
    )
    print(
        "Debt exceeds 100: "
        f"{format_optional_time(summary['selected_first_ell_above_100_time'])}"
    )
    print(
        "Debt exceeds 10,000: "
        f"{format_optional_time(summary['selected_first_ell_above_10000_time'])}"
    )

    print("\nTerminal stochastic debt-growth diagnostics")
    print("-" * 72)
    print(
        "Interest-growth gap A_t: "
        f"{summary['terminal_interest_growth_gap']:.10f}"
    )
    print(
        "Financing gap F_t: "
        f"{summary['terminal_financing_gap']:.10f}"
    )
    print(
        "Proportional financing F_t / ell_t: "
        f"{summary['terminal_proportional_financing']:.10e}"
    )
    print(
        "Ito correction -0.5 sigma_ell^2: "
        f"{summary['ito_log_debt_correction']:.10f}"
    )
    print(
        "Drift of log debt: "
        f"{summary['terminal_log_debt_drift']:.10f}"
    )


# =============================================================================
# Main workflow
# =============================================================================
def create_stochastic_crisis_figure(
    show: bool = True,
) -> tuple[plt.Figure, dict[str, object]]:
    """Run the ensemble, select the representative crisis, and plot it."""
    model = make_model()

    result = simulate_ensemble(model)
    selected_index = select_representative_crisis(result)
    selected_time, selected_states = extract_path(result, selected_index)
    selected_derived = calculate_path_derived(model, selected_states)
    summary = build_summary(model, result, selected_index)

    ensemble_csv, representative_csv, summary_json = save_outputs(
        result,
        selected_index,
        selected_time,
        selected_states,
        selected_derived,
        summary,
    )

    figure, png_path, pdf_path = create_main_figure(
        selected_time,
        selected_states,
        selected_derived,
        summary,
    )

    print_summary(summary)
    print(f"\nSaved figure:              {png_path}")
    print(f"Saved PDF:                 {pdf_path}")
    print(f"Saved ensemble data:       {ensemble_csv}")
    print(f"Saved representative path: {representative_csv}")
    print(f"Saved statistics:          {summary_json}")

    if show:
        plt.show()
    else:
        plt.close(figure)

    return figure, summary


if __name__ == "__main__":
    create_stochastic_crisis_figure(show=True)
