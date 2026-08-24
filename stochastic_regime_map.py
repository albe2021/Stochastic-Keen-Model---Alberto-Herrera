"""
stochastic_regime_map_single_slice_final.py

Final finite-horizon comparison between the deterministic regime map and
stochastic outcome probabilities at the low-debt equilibrium employment
slice.

The script answers the question: what happens to the deterministic low-debt
region when the baseline stochastic perturbations are switched on?

Design
------
* One fixed employment slice: lambda_0 = lambda_L^* ~= 0.6725.
* The deterministic map uses the same 500-year horizon, DOP853 solver,
  stopping thresholds, tail window, and convergence tolerances as the thesis
  deterministic regime map.
* At each initial condition, repeated unprojected Euler--Maruyama paths are
  simulated with baseline volatilities (0.05, 0.05, 0.05).
* The main figure shows:
    (a) deterministic finite-horizon classification;
    (b) stochastic probability of a low-debt outcome;
    (c) stochastic probability of a positive-debt crisis;
    (d) stochastic probability of a share-space exit.
* A supplementary figure shows the modal stochastic class and its probability.

The stochastic probabilities are finite-horizon probabilities under the
specified numerical design. They are not stochastic basins of attraction.
No clipping, projection, reflection, or epsilon floor is imposed.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Iterable, Mapping

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.integrate import solve_ivp

try:
    from keen_model_functions import KeenModel
except ImportError:
    KeenModel = None  # type: ignore[assignment,misc]


# =============================================================================
# 1. Experiment settings
# =============================================================================

# Use the low-debt equilibrium employment coordinate, matching the detailed
# deterministic regime map in the thesis.
LAMBDA_0: Final[float] = 0.6724861407

OMEGA_MIN: Final[float] = 0.20
OMEGA_MAX: Final[float] = 0.95
OMEGA_POINTS: Final[int] = 51

ELL_MIN: Final[float] = -1.0
ELL_MAX: Final[float] = 14.0
ELL_POINTS: Final[int] = 51

# Match the existing deterministic regime-map design.
T_MAP: Final[float] = 500.0
DT_STOCHASTIC: Final[float] = 0.05
TAIL_YEARS: Final[float] = 100.0
TAIL_OBSERVATIONS_DETERMINISTIC: Final[int] = 201

# Twenty paths keeps the run near the computational cost of the earlier
# five-slice experiment. Increase to 40 for the final Monte Carlo run if
# runtime permits.
N_STOCHASTIC_PATHS: Final[int] = 20
MASTER_SEED: Final[int] = 20260823
USE_COMMON_RANDOM_NUMBERS: Final[bool] = True

SIGMA: Final[np.ndarray] = np.array([0.05, 0.05, 0.05], dtype=float)

POSITIVE_DEBT_THRESHOLD: Final[float] = 50.0
NEGATIVE_DEBT_CUTOFF: Final[float] = -50.0

LOW_DEBT_SCALE: Final[float] = 3.0
HIGH_DEBT_SCALE: Final[float] = 6.0

# Deterministic convergence criteria: identical to the thesis regime maps.
DET_LOW_FINAL_TOL: Final[float] = 0.06
DET_LOW_TAIL_MEAN_TOL: Final[float] = 0.08
DET_LOW_TAIL_STD_TOL: Final[float] = 0.03

DET_HIGH_FINAL_TOL: Final[float] = 0.08
DET_HIGH_TAIL_MEAN_TOL: Final[float] = 0.12
DET_HIGH_TAIL_STD_TOL: Final[float] = 0.04

# Stochastic low-/high-debt outcome criteria. Exact convergence is not expected
# under persistent noise, so these tail tolerances are deliberately wider.
STOCH_LOW_FINAL_TOL: Final[float] = 0.25
STOCH_LOW_TAIL_MEAN_TOL: Final[float] = 0.30
STOCH_LOW_TAIL_STD_TOL: Final[float] = 0.15

STOCH_HIGH_FINAL_TOL: Final[float] = 0.35
STOCH_HIGH_TAIL_MEAN_TOL: Final[float] = 0.40
STOCH_HIGH_TAIL_STD_TOL: Final[float] = 0.20

# Deterministic DOP853 settings, matching the thesis maps.
DET_RTOL: Final[float] = 1.0e-8
DET_ATOL: Final[float] = 1.0e-10
DET_MAX_STEP: Final[float] = 0.5

OUTPUT_DIRECTORY: Final[Path] = Path("regime_map_single_slice_outputs")

MAIN_FIGURE_NAME: Final[str] = "deterministic_stochastic_regime_probabilities_lambda_low.png"
MODAL_FIGURE_NAME: Final[str] = "stochastic_regime_modal_diagnostic_lambda_low.png"
CELL_DATA_NAME: Final[str] = "stochastic_regime_probability_cells_lambda_low.csv"
SUMMARY_JSON_NAME: Final[str] = "stochastic_regime_probability_summary_lambda_low.json"
SUMMARY_REPORT_NAME: Final[str] = "stochastic_regime_probability_report_lambda_low.txt"


# =============================================================================
# 2. Calibration and reference points
# =============================================================================

LOW_DEBT_EQUILIBRIUM: Final[np.ndarray] = np.array(
    [0.6276666667, 0.6724861407, 0.0166666667], dtype=float
)
HIGH_DEBT_EQUILIBRIUM: Final[np.ndarray] = np.array(
    [0.3920000000, 0.6543965885, 11.8000000000], dtype=float
)

FAVOURABLE_INITIAL_PROJECTION: Final[tuple[float, float]] = (0.9, 0.3)
UNFAVOURABLE_INITIAL_PROJECTION: Final[tuple[float, float]] = (0.578, 1.53)

FALLBACK_BASELINE_PARAMS: Final[dict[str, float]] = {
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


def _first_available(
    source: Mapping[str, float], names: Iterable[str], default: float
) -> float:
    for name in names:
        if name in source:
            return float(source[name])
    return float(default)


def load_baseline_parameters() -> dict[str, float]:
    """Load the canonical calibration, falling back to embedded values."""
    parameters = FALLBACK_BASELINE_PARAMS.copy()

    if KeenModel is not None:
        try:
            if hasattr(KeenModel, "DEFAULT_PARAMS"):
                source = dict(KeenModel.DEFAULT_PARAMS)
            else:
                source = dict(KeenModel().params)

            for key in (
                "alpha",
                "beta",
                "delta",
                "nu",
                "r",
                "eta_p",
                "m",
                "gamma",
                "phi0",
                "phi1",
                "kappa0",
                "kappa1",
                "kappa_min",
                "kappa_max",
            ):
                if key in source:
                    parameters[key] = float(source[key])

            parameters["dividend0"] = _first_available(
                source, ("dividend0", "delta0"), parameters["dividend0"]
            )
            parameters["dividend1"] = _first_available(
                source, ("dividend1", "delta1"), parameters["dividend1"]
            )
            parameters["dividend_min"] = _first_available(
                source,
                ("dividend_min", "delta_min"),
                parameters["dividend_min"],
            )
            parameters["dividend_max"] = _first_available(
                source,
                ("dividend_max", "delta_max"),
                parameters["dividend_max"],
            )
        except Exception as exc:
            print(
                "Warning: could not read KeenModel defaults; "
                f"using embedded calibration ({exc})."
            )

    parameters["sigma_omega"] = float(SIGMA[0])
    parameters["sigma_lambda"] = float(SIGMA[1])
    parameters["sigma_ell"] = float(SIGMA[2])
    return parameters


# =============================================================================
# 3. Outcome classes and plotting colours
# =============================================================================

LOW_DEBT: Final[int] = 0
HIGH_DEBT: Final[int] = 1
POSITIVE_DEBT_CRISIS: Final[int] = 2
SHARE_SPACE_EXIT: Final[int] = 3
NEGATIVE_DEBT_EXIT: Final[int] = 4
UNRESOLVED: Final[int] = 5
NUMERICAL_FAILURE: Final[int] = 6

CLASS_CODES: Final[tuple[int, ...]] = (
    LOW_DEBT,
    HIGH_DEBT,
    POSITIVE_DEBT_CRISIS,
    SHARE_SPACE_EXIT,
    NEGATIVE_DEBT_EXIT,
    UNRESOLVED,
    NUMERICAL_FAILURE,
)

CLASS_LABELS: Final[dict[int, str]] = {
    LOW_DEBT: "Low-debt outcome",
    HIGH_DEBT: "High-debt outcome",
    POSITIVE_DEBT_CRISIS: "Positive-debt crisis",
    SHARE_SPACE_EXIT: "Share-space exit",
    NEGATIVE_DEBT_EXIT: "Negative-debt cutoff",
    UNRESOLVED: "Unresolved",
    NUMERICAL_FAILURE: "Numerical failure",
}

CLASS_SHORT_NAMES: Final[dict[int, str]] = {
    LOW_DEBT: "low_debt",
    HIGH_DEBT: "high_debt",
    POSITIVE_DEBT_CRISIS: "positive_debt_crisis",
    SHARE_SPACE_EXIT: "share_space_exit",
    NEGATIVE_DEBT_EXIT: "negative_debt_cutoff",
    UNRESOLVED: "unresolved",
    NUMERICAL_FAILURE: "numerical_failure",
}

# Exact thesis deterministic-map palette.
PLOT_COLORS: Final[list[str]] = [
    "#d9d9d9",  # unresolved
    "#969696",  # negative-debt safety cutoff
    "#D2F9DB",  # share-space exit
    "#e57373",  # positive-debt crisis
    "#63E981",  # low-debt convergence
    "#f6c16b",  # high-debt convergence
]

CLASS_COLOURS: Final[dict[int, str]] = {
    LOW_DEBT: PLOT_COLORS[4],
    HIGH_DEBT: PLOT_COLORS[5],
    POSITIVE_DEBT_CRISIS: PLOT_COLORS[3],
    SHARE_SPACE_EXIT: PLOT_COLORS[2],
    NEGATIVE_DEBT_EXIT: PLOT_COLORS[1],
    UNRESOLVED: PLOT_COLORS[0],
    NUMERICAL_FAILURE: "#222222",
}

REGIME_CMAP = ListedColormap([CLASS_COLOURS[code] for code in CLASS_CODES])
REGIME_NORM = BoundaryNorm(
    np.arange(-0.5, len(CLASS_CODES) + 0.5, 1.0), REGIME_CMAP.N
)

LOW_PROBABILITY_CMAP = LinearSegmentedColormap.from_list(
    "low_probability", ["#ffffff", CLASS_COLOURS[LOW_DEBT]]
)
CRISIS_PROBABILITY_CMAP = LinearSegmentedColormap.from_list(
    "crisis_probability", ["#ffffff", CLASS_COLOURS[POSITIVE_DEBT_CRISIS]]
)
EXIT_PROBABILITY_CMAP = LinearSegmentedColormap.from_list(
    "exit_probability", ["#ffffff", "#69c783"]
)


# =============================================================================
# 4. Model equations
# =============================================================================


def model_terms(
    omega: np.ndarray,
    lam: np.ndarray,
    ell: np.ndarray,
    parameters: Mapping[str, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the drift and diffusion amplitudes, vectorised over arrays."""
    profit = 1.0 - omega - parameters["r"] * ell

    investment = np.clip(
        parameters["kappa0"] + parameters["kappa1"] * profit,
        parameters["kappa_min"],
        parameters["kappa_max"],
    )
    dividend = np.clip(
        parameters["dividend0"] + parameters["dividend1"] * profit,
        parameters["dividend_min"],
        parameters["dividend_max"],
    )
    inflation = parameters["eta_p"] * (parameters["m"] * omega - 1.0)
    phillips = parameters["phi0"] + parameters["phi1"] * lam

    drift_omega = omega * (
        phillips
        - parameters["alpha"]
        - (1.0 - parameters["gamma"]) * inflation
    )
    drift_lambda = lam * (
        investment / parameters["nu"]
        - parameters["delta"]
        - parameters["alpha"]
        - parameters["beta"]
    )
    drift_ell = (
        ell
        * (
            parameters["r"]
            - investment / parameters["nu"]
            + parameters["delta"]
            - inflation
        )
        + omega
        + investment
        - 1.0
        + dividend
    )

    sigma_omega = parameters["sigma_omega"] * np.sqrt(
        np.maximum(omega * (1.0 - omega), 0.0)
    )
    sigma_lambda = parameters["sigma_lambda"] * np.sqrt(
        np.maximum(lam * (1.0 - lam), 0.0)
    )
    sigma_ell = parameters["sigma_ell"] * np.abs(ell)

    return (
        drift_omega,
        drift_lambda,
        drift_ell,
        sigma_omega,
        sigma_lambda,
        sigma_ell,
    )


def deterministic_rhs(
    _time: float, state: np.ndarray, parameters: Mapping[str, float]
) -> np.ndarray:
    omega, lam, ell = np.asarray(state, dtype=float)
    terms = model_terms(
        np.asarray(omega), np.asarray(lam), np.asarray(ell), parameters
    )
    return np.array([terms[0], terms[1], terms[2]], dtype=float)


# =============================================================================
# 5. Grid and classification helpers
# =============================================================================


@dataclass(frozen=True)
class TailTolerances:
    low_final: float
    low_mean: float
    low_std: float
    high_final: float
    high_mean: float
    high_std: float


DETERMINISTIC_TOLERANCES = TailTolerances(
    low_final=DET_LOW_FINAL_TOL,
    low_mean=DET_LOW_TAIL_MEAN_TOL,
    low_std=DET_LOW_TAIL_STD_TOL,
    high_final=DET_HIGH_FINAL_TOL,
    high_mean=DET_HIGH_TAIL_MEAN_TOL,
    high_std=DET_HIGH_TAIL_STD_TOL,
)

STOCHASTIC_TOLERANCES = TailTolerances(
    low_final=STOCH_LOW_FINAL_TOL,
    low_mean=STOCH_LOW_TAIL_MEAN_TOL,
    low_std=STOCH_LOW_TAIL_STD_TOL,
    high_final=STOCH_HIGH_FINAL_TOL,
    high_mean=STOCH_HIGH_TAIL_MEAN_TOL,
    high_std=STOCH_HIGH_TAIL_STD_TOL,
)


def validate_settings() -> None:
    if T_MAP <= 0.0 or DT_STOCHASTIC <= 0.0:
        raise ValueError("T_MAP and DT_STOCHASTIC must be positive.")
    ratio = T_MAP / DT_STOCHASTIC
    if not np.isclose(ratio, round(ratio), rtol=0.0, atol=1.0e-10):
        raise ValueError("T_MAP / DT_STOCHASTIC must be an integer.")
    if TAIL_YEARS <= 0.0 or TAIL_YEARS >= T_MAP:
        raise ValueError("TAIL_YEARS must lie strictly between 0 and T_MAP.")
    if OMEGA_POINTS < 2 or ELL_POINTS < 2:
        raise ValueError("Each grid dimension needs at least two points.")
    if N_STOCHASTIC_PATHS < 1:
        raise ValueError("N_STOCHASTIC_PATHS must be at least one.")
    if POSITIVE_DEBT_THRESHOLD <= ELL_MAX:
        raise ValueError("Positive-debt threshold must exceed ELL_MAX.")
    if NEGATIVE_DEBT_CUTOFF >= ELL_MIN:
        raise ValueError("Negative-debt cutoff must be below ELL_MIN.")


def make_initial_grid() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    omega_grid = np.linspace(OMEGA_MIN, OMEGA_MAX, OMEGA_POINTS)
    ell_grid = np.linspace(ELL_MIN, ELL_MAX, ELL_POINTS)
    ell_mesh, omega_mesh = np.meshgrid(ell_grid, omega_grid)
    states = np.column_stack(
        [
            omega_mesh.ravel(),
            np.full(omega_mesh.size, LAMBDA_0, dtype=float),
            ell_mesh.ravel(),
        ]
    )
    return states, omega_grid, ell_grid


def scaled_distance(
    omega: np.ndarray,
    lam: np.ndarray,
    ell: np.ndarray,
    equilibrium: np.ndarray,
    debt_scale: float,
) -> np.ndarray:
    return np.sqrt(
        (omega - equilibrium[0]) ** 2
        + (lam - equilibrium[1]) ** 2
        + ((ell - equilibrium[2]) / debt_scale) ** 2
    )


def classify_tail_statistics(
    omega_final: np.ndarray,
    lambda_final: np.ndarray,
    ell_final: np.ndarray,
    low_mean: np.ndarray,
    low_std: np.ndarray,
    high_mean: np.ndarray,
    high_std: np.ndarray,
    tolerances: TailTolerances,
) -> np.ndarray:
    classes = np.full(np.shape(omega_final), UNRESOLVED, dtype=np.int8)

    low_final = scaled_distance(
        omega_final,
        lambda_final,
        ell_final,
        LOW_DEBT_EQUILIBRIUM,
        LOW_DEBT_SCALE,
    )
    high_final = scaled_distance(
        omega_final,
        lambda_final,
        ell_final,
        HIGH_DEBT_EQUILIBRIUM,
        HIGH_DEBT_SCALE,
    )

    low_condition = (
        (low_final < tolerances.low_final)
        & (low_mean < tolerances.low_mean)
        & (low_std < tolerances.low_std)
    )
    high_condition = (
        (high_final < tolerances.high_final)
        & (high_mean < tolerances.high_mean)
        & (high_std < tolerances.high_std)
    )

    only_low = low_condition & ~high_condition
    only_high = high_condition & ~low_condition
    both = low_condition & high_condition

    classes[only_low] = LOW_DEBT
    classes[only_high] = HIGH_DEBT
    classes[both & (low_mean <= high_mean)] = LOW_DEBT
    classes[both & (high_mean < low_mean)] = HIGH_DEBT
    return classes


def assign_stochastic_terminal_events(
    omega: np.ndarray,
    lam: np.ndarray,
    ell: np.ndarray,
    active: np.ndarray,
    classes: np.ndarray,
) -> None:
    finite = np.isfinite(omega) & np.isfinite(lam) & np.isfinite(ell)
    failure = active & ~finite
    classes[failure] = NUMERICAL_FAILURE
    active[failure] = False

    positive_crisis = active & (ell >= POSITIVE_DEBT_THRESHOLD)
    classes[positive_crisis] = POSITIVE_DEBT_CRISIS
    active[positive_crisis] = False

    negative_cutoff = active & (ell <= NEGATIVE_DEBT_CUTOFF)
    classes[negative_cutoff] = NEGATIVE_DEBT_EXIT
    active[negative_cutoff] = False

    share_exit = active & (
        (omega <= 0.0)
        | (omega >= 1.0)
        | (lam <= 0.0)
        | (lam >= 1.0)
    )
    classes[share_exit] = SHARE_SPACE_EXIT
    active[share_exit] = False


# =============================================================================
# 6. Deterministic map: DOP853 and original thesis criteria
# =============================================================================


@dataclass
class DeterministicResult:
    omega_grid: np.ndarray
    ell_grid: np.ndarray
    classes: np.ndarray


def _event_omega_zero(_t: float, y: np.ndarray, _p: Mapping[str, float]) -> float:
    return float(y[0])


def _event_omega_one(_t: float, y: np.ndarray, _p: Mapping[str, float]) -> float:
    return float(1.0 - y[0])


def _event_lambda_zero(_t: float, y: np.ndarray, _p: Mapping[str, float]) -> float:
    return float(y[1])


def _event_lambda_one(_t: float, y: np.ndarray, _p: Mapping[str, float]) -> float:
    return float(1.0 - y[1])


def _event_positive_debt(_t: float, y: np.ndarray, _p: Mapping[str, float]) -> float:
    return float(POSITIVE_DEBT_THRESHOLD - y[2])


def _event_negative_debt(_t: float, y: np.ndarray, _p: Mapping[str, float]) -> float:
    return float(y[2] - NEGATIVE_DEBT_CUTOFF)


for _event in (
    _event_omega_zero,
    _event_omega_one,
    _event_lambda_zero,
    _event_lambda_one,
    _event_positive_debt,
    _event_negative_debt,
):
    _event.terminal = True  # type: ignore[attr-defined]
    _event.direction = -1.0  # type: ignore[attr-defined]


def classify_deterministic_initial_state(
    initial_state: np.ndarray,
    parameters: Mapping[str, float],
) -> int:
    deterministic_parameters = dict(parameters)
    deterministic_parameters["sigma_omega"] = 0.0
    deterministic_parameters["sigma_lambda"] = 0.0
    deterministic_parameters["sigma_ell"] = 0.0

    events = (
        _event_omega_zero,
        _event_omega_one,
        _event_lambda_zero,
        _event_lambda_one,
        _event_positive_debt,
        _event_negative_debt,
    )

    solution = solve_ivp(
        fun=deterministic_rhs,
        t_span=(0.0, T_MAP),
        y0=np.asarray(initial_state, dtype=float),
        method="DOP853",
        rtol=DET_RTOL,
        atol=DET_ATOL,
        max_step=DET_MAX_STEP,
        dense_output=True,
        events=events,
        args=(deterministic_parameters,),
    )

    if not solution.success:
        return NUMERICAL_FAILURE

    # Find the earliest event, if any. Event indices 0--3 are share exits,
    # index 4 is a positive-debt crisis, and index 5 is the negative cutoff.
    event_times: list[tuple[float, int]] = []
    for event_index, times in enumerate(solution.t_events):
        if len(times) > 0:
            event_times.append((float(times[0]), event_index))

    if event_times:
        _, event_index = min(event_times, key=lambda item: item[0])
        if event_index <= 3:
            return SHARE_SPACE_EXIT
        if event_index == 4:
            return POSITIVE_DEBT_CRISIS
        return NEGATIVE_DEBT_EXIT

    if solution.sol is None:
        return NUMERICAL_FAILURE

    tail_times = np.linspace(
        T_MAP - TAIL_YEARS,
        T_MAP,
        TAIL_OBSERVATIONS_DETERMINISTIC,
    )
    tail_states = solution.sol(tail_times).T
    if not np.all(np.isfinite(tail_states)):
        return NUMERICAL_FAILURE

    low_distances = scaled_distance(
        tail_states[:, 0],
        tail_states[:, 1],
        tail_states[:, 2],
        LOW_DEBT_EQUILIBRIUM,
        LOW_DEBT_SCALE,
    )
    high_distances = scaled_distance(
        tail_states[:, 0],
        tail_states[:, 1],
        tail_states[:, 2],
        HIGH_DEBT_EQUILIBRIUM,
        HIGH_DEBT_SCALE,
    )

    result = classify_tail_statistics(
        np.asarray([tail_states[-1, 0]]),
        np.asarray([tail_states[-1, 1]]),
        np.asarray([tail_states[-1, 2]]),
        np.asarray([np.mean(low_distances)]),
        np.asarray([np.std(low_distances, ddof=1)]),
        np.asarray([np.mean(high_distances)]),
        np.asarray([np.std(high_distances, ddof=1)]),
        DETERMINISTIC_TOLERANCES,
    )
    return int(result[0])


def simulate_deterministic_map(
    parameters: Mapping[str, float],
) -> DeterministicResult:
    states, omega_grid, ell_grid = make_initial_grid()
    classes = np.full(len(states), UNRESOLVED, dtype=np.int8)

    print("Computing deterministic DOP853 map")
    for index, initial_state in enumerate(states):
        classes[index] = classify_deterministic_initial_state(
            initial_state, parameters
        )
        if (index + 1) % max(1, len(states) // 10) == 0:
            print(f"  deterministic cells: {index + 1}/{len(states)}")

    return DeterministicResult(
        omega_grid=omega_grid,
        ell_grid=ell_grid,
        classes=classes.reshape(OMEGA_POINTS, ELL_POINTS),
    )


# =============================================================================
# 7. Stochastic probability map
# =============================================================================


@dataclass
class StochasticResult:
    omega_grid: np.ndarray
    ell_grid: np.ndarray
    modal_class: np.ndarray
    modal_probability: np.ndarray
    class_probabilities: np.ndarray


def _normal_shocks(
    rng: np.random.Generator,
    n_cells: int,
    n_paths: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if USE_COMMON_RANDOM_NUMBERS:
        shocks = rng.standard_normal((3, 1, n_paths))
        return (
            np.broadcast_to(shocks[0], (n_cells, n_paths)),
            np.broadcast_to(shocks[1], (n_cells, n_paths)),
            np.broadcast_to(shocks[2], (n_cells, n_paths)),
        )
    shocks = rng.standard_normal((3, n_cells, n_paths))
    return shocks[0], shocks[1], shocks[2]


def simulate_stochastic_map(
    parameters: Mapping[str, float],
) -> StochasticResult:
    initial_states, omega_grid, ell_grid = make_initial_grid()
    n_cells = len(initial_states)
    n_paths = N_STOCHASTIC_PATHS

    omega = np.repeat(initial_states[:, 0, None], n_paths, axis=1)
    lam = np.repeat(initial_states[:, 1, None], n_paths, axis=1)
    ell = np.repeat(initial_states[:, 2, None], n_paths, axis=1)

    classes = np.full((n_cells, n_paths), UNRESOLVED, dtype=np.int8)
    active = np.ones((n_cells, n_paths), dtype=bool)

    low_sum = np.zeros((n_cells, n_paths), dtype=float)
    low_sumsq = np.zeros((n_cells, n_paths), dtype=float)
    high_sum = np.zeros((n_cells, n_paths), dtype=float)
    high_sumsq = np.zeros((n_cells, n_paths), dtype=float)
    tail_count = np.zeros((n_cells, n_paths), dtype=np.int32)

    n_steps = int(round(T_MAP / DT_STOCHASTIC))
    tail_start_step = int(round((T_MAP - TAIL_YEARS) / DT_STOCHASTIC))
    sqrt_dt = np.sqrt(DT_STOCHASTIC)
    rng = np.random.default_rng(np.random.SeedSequence([MASTER_SEED, 6725]))

    print("Computing stochastic probability map")
    for step in range(1, n_steps + 1):
        if not np.any(active):
            break

        active_omega = omega[active]
        active_lambda = lam[active]
        active_ell = ell[active]
        terms = model_terms(
            active_omega, active_lambda, active_ell, parameters
        )
        z_omega, z_lambda, z_ell = _normal_shocks(rng, n_cells, n_paths)

        omega[active] = (
            active_omega
            + terms[0] * DT_STOCHASTIC
            + terms[3] * sqrt_dt * z_omega[active]
        )
        lam[active] = (
            active_lambda
            + terms[1] * DT_STOCHASTIC
            + terms[4] * sqrt_dt * z_lambda[active]
        )
        ell[active] = (
            active_ell
            + terms[2] * DT_STOCHASTIC
            + terms[5] * sqrt_dt * z_ell[active]
        )

        assign_stochastic_terminal_events(omega, lam, ell, active, classes)

        if step >= tail_start_step and np.any(active):
            low_distance = scaled_distance(
                omega[active],
                lam[active],
                ell[active],
                LOW_DEBT_EQUILIBRIUM,
                LOW_DEBT_SCALE,
            )
            high_distance = scaled_distance(
                omega[active],
                lam[active],
                ell[active],
                HIGH_DEBT_EQUILIBRIUM,
                HIGH_DEBT_SCALE,
            )
            low_sum[active] += low_distance
            low_sumsq[active] += low_distance**2
            high_sum[active] += high_distance
            high_sumsq[active] += high_distance**2
            tail_count[active] += 1

        if step % max(1, n_steps // 10) == 0:
            print(
                f"  stochastic step {step}/{n_steps}; "
                f"active paths={int(np.sum(active))}"
            )

    if np.any(active):
        valid_tail = active & (tail_count > 0)

        low_mean = np.full_like(omega, np.inf, dtype=float)
        high_mean = np.full_like(omega, np.inf, dtype=float)
        low_std = np.full_like(omega, np.inf, dtype=float)
        high_std = np.full_like(omega, np.inf, dtype=float)

        low_mean[valid_tail] = low_sum[valid_tail] / tail_count[valid_tail]
        high_mean[valid_tail] = high_sum[valid_tail] / tail_count[valid_tail]
        low_variance = np.maximum(
            low_sumsq[valid_tail] / tail_count[valid_tail]
            - low_mean[valid_tail] ** 2,
            0.0,
        )
        high_variance = np.maximum(
            high_sumsq[valid_tail] / tail_count[valid_tail]
            - high_mean[valid_tail] ** 2,
            0.0,
        )
        low_std[valid_tail] = np.sqrt(low_variance)
        high_std[valid_tail] = np.sqrt(high_variance)

        completed_classes = classify_tail_statistics(
            omega[valid_tail],
            lam[valid_tail],
            ell[valid_tail],
            low_mean[valid_tail],
            low_std[valid_tail],
            high_mean[valid_tail],
            high_std[valid_tail],
            STOCHASTIC_TOLERANCES,
        )
        classes[valid_tail] = completed_classes

    class_counts = np.stack(
        [np.sum(classes == code, axis=1) for code in CLASS_CODES], axis=1
    )
    modal_index = np.argmax(class_counts, axis=1)
    modal_class = np.asarray(CLASS_CODES, dtype=np.int8)[modal_index]
    modal_probability = np.max(class_counts, axis=1) / float(n_paths)
    class_probabilities = class_counts.astype(float) / float(n_paths)

    return StochasticResult(
        omega_grid=omega_grid,
        ell_grid=ell_grid,
        modal_class=modal_class.reshape(OMEGA_POINTS, ELL_POINTS),
        modal_probability=modal_probability.reshape(OMEGA_POINTS, ELL_POINTS),
        class_probabilities=class_probabilities.reshape(
            OMEGA_POINTS, ELL_POINTS, len(CLASS_CODES)
        ),
    )


# =============================================================================
# 8. Plotting
# =============================================================================


plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)


def add_reference_markers(ax: plt.Axes) -> None:
    ax.plot(
        LOW_DEBT_EQUILIBRIUM[2],
        LOW_DEBT_EQUILIBRIUM[0],
        marker="*",
        markersize=12,
        color="black",
        linestyle="None",
        zorder=7,
    )
    ax.plot(
        HIGH_DEBT_EQUILIBRIUM[2],
        HIGH_DEBT_EQUILIBRIUM[0],
        marker="D",
        markersize=7,
        color="black",
        linestyle="None",
        zorder=7,
    )
    ax.plot(
        FAVOURABLE_INITIAL_PROJECTION[1],
        FAVOURABLE_INITIAL_PROJECTION[0],
        marker="o",
        markersize=6,
        markerfacecolor="white",
        markeredgecolor="black",
        linestyle="None",
        zorder=7,
    )
    ax.plot(
        UNFAVOURABLE_INITIAL_PROJECTION[1],
        UNFAVOURABLE_INITIAL_PROJECTION[0],
        marker="s",
        markersize=6,
        markerfacecolor="white",
        markeredgecolor="black",
        linestyle="None",
        zorder=7,
    )


def add_common_axes_format(ax: plt.Axes, title: str) -> None:
    ax.set_title(title, fontweight="bold")
    ax.set_xlim(ELL_MIN, ELL_MAX)
    ax.set_ylim(OMEGA_MIN, OMEGA_MAX)
    ax.set_xlabel(r"Initial debt $\ell_0$")
    ax.set_ylabel(r"Initial wage share $\omega_0$")
    ax.grid(True, alpha=0.20)
    add_reference_markers(ax)


def deterministic_low_boundary(
    ax: plt.Axes,
    deterministic: DeterministicResult,
    colour: str = "blue",
    linewidth: float = 1.3,
) -> None:
    ell_mesh, omega_mesh = np.meshgrid(
        deterministic.ell_grid, deterministic.omega_grid
    )
    low_mask = (deterministic.classes == LOW_DEBT).astype(float)
    if np.min(low_mask) < 0.5 < np.max(low_mask):
        ax.contour(
            ell_mesh,
            omega_mesh,
            low_mask,
            levels=[0.5],
            colors=colour,
            linewidths=linewidth,
        )


def plot_probability_panel(
    ax: plt.Axes,
    deterministic: DeterministicResult,
    stochastic: StochasticResult,
    probability: np.ndarray,
    title: str,
    cmap: LinearSegmentedColormap,
) -> object:
    ell_mesh, omega_mesh = np.meshgrid(
        stochastic.ell_grid, stochastic.omega_grid
    )
    image = ax.pcolormesh(
        ell_mesh,
        omega_mesh,
        probability,
        cmap=cmap,
        norm=Normalize(vmin=0.0, vmax=1.0),
        shading="auto",
    )
    deterministic_low_boundary(ax, deterministic)
    add_common_axes_format(ax, title)
    return image


def create_main_figure(
    deterministic: DeterministicResult,
    stochastic: StochasticResult,
) -> Path:
    figure, axes = plt.subplots(2, 2, figsize=(13.5, 10.0))

    ell_mesh, omega_mesh = np.meshgrid(
        deterministic.ell_grid, deterministic.omega_grid
    )
    axes[0, 0].pcolormesh(
        ell_mesh,
        omega_mesh,
        deterministic.classes,
        cmap=REGIME_CMAP,
        norm=REGIME_NORM,
        shading="auto",
        alpha=0.95,
    )
    deterministic_low_boundary(axes[0, 0], deterministic)
    add_common_axes_format(
        axes[0, 0],
        rf"(a) Deterministic classification, $\lambda_0={LAMBDA_0:.4f}$",
    )

    low_index = CLASS_CODES.index(LOW_DEBT)
    crisis_index = CLASS_CODES.index(POSITIVE_DEBT_CRISIS)
    exit_index = CLASS_CODES.index(SHARE_SPACE_EXIT)

    low_image = plot_probability_panel(
        axes[0, 1],
        deterministic,
        stochastic,
        stochastic.class_probabilities[:, :, low_index],
        "(b) Probability of a low-debt outcome",
        LOW_PROBABILITY_CMAP,
    )
    crisis_image = plot_probability_panel(
        axes[1, 0],
        deterministic,
        stochastic,
        stochastic.class_probabilities[:, :, crisis_index],
        "(c) Probability of a positive-debt crisis",
        CRISIS_PROBABILITY_CMAP,
    )
    exit_image = plot_probability_panel(
        axes[1, 1],
        deterministic,
        stochastic,
        stochastic.class_probabilities[:, :, exit_index],
        "(d) Probability of a share-space exit",
        EXIT_PROBABILITY_CMAP,
    )

    for ax, image in (
        (axes[0, 1], low_image),
        (axes[1, 0], crisis_image),
        (axes[1, 1], exit_image),
    ):
        colour_bar = figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        colour_bar.set_label("Finite-horizon probability")

    legend_handles: list[object] = [
        Patch(
            facecolor=CLASS_COLOURS[code],
            edgecolor="none",
            label=CLASS_LABELS[code],
        )
        for code in CLASS_CODES
    ]
    legend_handles.extend(
        [
            Line2D(
                [0],
                [0],
                color="blue",
                linewidth=1.4,
                label="Deterministic low-debt boundary",
            ),
            Line2D(
                [0],
                [0],
                marker="*",
                color="black",
                linestyle="None",
                markersize=11,
                label="Low-debt equilibrium projection",
            ),
            Line2D(
                [0],
                [0],
                marker="D",
                color="black",
                linestyle="None",
                markersize=7,
                label="High-debt equilibrium projection",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                markerfacecolor="white",
                markeredgecolor="black",
                color="none",
                linestyle="None",
                markersize=6,
                label="Favourable initial-condition projection",
            ),
            Line2D(
                [0],
                [0],
                marker="s",
                markerfacecolor="white",
                markeredgecolor="black",
                color="none",
                linestyle="None",
                markersize=6,
                label="Unfavourable initial-condition projection",
            ),
        ]
    )

    figure.suptitle(
        "Deterministic Classification and Stochastic Outcome Probabilities\n"
        rf"$\lambda_0={LAMBDA_0:.4f}$; baseline volatilities; "
        rf"{N_STOCHASTIC_PATHS} stochastic paths per initial condition",
        fontsize=16,
        fontweight="bold",
        y=0.99,
    )
    figure.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.01),
        frameon=True,
    )
    figure.subplots_adjust(
        left=0.07,
        right=0.97,
        top=0.90,
        bottom=0.14,
        wspace=0.24,
        hspace=0.26,
    )

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIRECTORY / MAIN_FIGURE_NAME
    figure.savefig(path)
    return path


def create_modal_diagnostic_figure(
    deterministic: DeterministicResult,
    stochastic: StochasticResult,
) -> Path:
    figure, axes = plt.subplots(1, 2, figsize=(13.0, 5.0))
    ell_mesh, omega_mesh = np.meshgrid(
        stochastic.ell_grid, stochastic.omega_grid
    )

    axes[0].pcolormesh(
        ell_mesh,
        omega_mesh,
        stochastic.modal_class,
        cmap=REGIME_CMAP,
        norm=REGIME_NORM,
        shading="auto",
        alpha=0.95,
    )
    deterministic_low_boundary(axes[0], deterministic)
    add_common_axes_format(axes[0], "(a) Modal stochastic class")

    probability_image = axes[1].pcolormesh(
        ell_mesh,
        omega_mesh,
        stochastic.modal_probability,
        cmap="viridis",
        norm=Normalize(vmin=0.0, vmax=1.0),
        shading="auto",
    )
    axes[1].contour(
        ell_mesh,
        omega_mesh,
        stochastic.modal_probability,
        levels=[0.60, 0.80],
        colors=["white", "black"],
        linewidths=[1.0, 1.0],
    )
    deterministic_low_boundary(axes[1], deterministic, colour="cyan")
    add_common_axes_format(axes[1], "(b) Probability of the modal class")

    colour_bar = figure.colorbar(
        probability_image, ax=axes[1], fraction=0.046, pad=0.04
    )
    colour_bar.set_label("Modal outcome probability")

    axes[1].legend(
        handles=[
            Line2D([0], [0], color="white", linewidth=1.5, label=r"$p_{\rm modal}=0.60$"),
            Line2D([0], [0], color="black", linewidth=1.5, label=r"$p_{\rm modal}=0.80$"),
            Line2D([0], [0], color="cyan", linewidth=1.5, label="Deterministic low-debt boundary"),
        ],
        loc="lower right",
        framealpha=0.95,
    )

    figure.suptitle(
        "Supplementary Stochastic Regime-Map Diagnostics",
        fontsize=15,
        fontweight="bold",
    )
    figure.subplots_adjust(
        left=0.07,
        right=0.97,
        top=0.87,
        bottom=0.13,
        wspace=0.22,
    )

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIRECTORY / MODAL_FIGURE_NAME
    figure.savefig(path)
    return path


# =============================================================================
# 9. Exports and comparison metrics
# =============================================================================


def nearest_grid_indices(
    omega_grid: np.ndarray,
    ell_grid: np.ndarray,
    omega_value: float,
    ell_value: float,
) -> tuple[int, int]:
    return (
        int(np.argmin(np.abs(omega_grid - omega_value))),
        int(np.argmin(np.abs(ell_grid - ell_value))),
    )


def calculate_summary(
    deterministic: DeterministicResult,
    stochastic: StochasticResult,
) -> dict[str, object]:
    probabilities = stochastic.class_probabilities
    low_probability = probabilities[:, :, CLASS_CODES.index(LOW_DEBT)]
    crisis_probability = probabilities[
        :, :, CLASS_CODES.index(POSITIVE_DEBT_CRISIS)
    ]
    exit_probability = probabilities[:, :, CLASS_CODES.index(SHARE_SPACE_EXIT)]
    unresolved_probability = probabilities[:, :, CLASS_CODES.index(UNRESOLVED)]

    det_low_mask = deterministic.classes == LOW_DEBT
    det_nonlow_mask = ~det_low_mask

    def masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
        return float(np.mean(values[mask])) if np.any(mask) else float("nan")

    def masked_fraction(condition: np.ndarray, mask: np.ndarray) -> float:
        return float(np.mean(condition[mask])) if np.any(mask) else float("nan")

    favourable_index = nearest_grid_indices(
        stochastic.omega_grid,
        stochastic.ell_grid,
        *FAVOURABLE_INITIAL_PROJECTION,
    )
    unfavourable_index = nearest_grid_indices(
        stochastic.omega_grid,
        stochastic.ell_grid,
        *UNFAVOURABLE_INITIAL_PROJECTION,
    )

    summary: dict[str, object] = {
        "settings": {
            "lambda0": LAMBDA_0,
            "omega_points": OMEGA_POINTS,
            "ell_points": ELL_POINTS,
            "horizon": T_MAP,
            "stochastic_dt": DT_STOCHASTIC,
            "tail_years": TAIL_YEARS,
            "paths_per_cell": N_STOCHASTIC_PATHS,
            "master_seed": MASTER_SEED,
            "volatility_vector": SIGMA.tolist(),
            "positive_debt_threshold": POSITIVE_DEBT_THRESHOLD,
            "negative_debt_cutoff": NEGATIVE_DEBT_CUTOFF,
        },
        "deterministic": {
            CLASS_SHORT_NAMES[code]: int(np.sum(deterministic.classes == code))
            for code in CLASS_CODES
        },
        "stochastic_grid_means": {
            "low_debt_probability": float(np.mean(low_probability)),
            "positive_debt_crisis_probability": float(np.mean(crisis_probability)),
            "share_space_exit_probability": float(np.mean(exit_probability)),
            "unresolved_probability": float(np.mean(unresolved_probability)),
            "modal_probability": float(np.mean(stochastic.modal_probability)),
        },
        "inside_deterministic_low_debt_region": {
            "cells": int(np.sum(det_low_mask)),
            "mean_low_debt_probability": masked_mean(low_probability, det_low_mask),
            "mean_positive_crisis_probability": masked_mean(
                crisis_probability, det_low_mask
            ),
            "mean_share_exit_probability": masked_mean(exit_probability, det_low_mask),
            "fraction_cells_p_low_at_least_0_50": masked_fraction(
                low_probability >= 0.50, det_low_mask
            ),
            "fraction_cells_p_low_at_least_0_75": masked_fraction(
                low_probability >= 0.75, det_low_mask
            ),
        },
        "outside_deterministic_low_debt_region": {
            "cells": int(np.sum(det_nonlow_mask)),
            "mean_low_debt_probability": masked_mean(
                low_probability, det_nonlow_mask
            ),
            "mean_positive_crisis_probability": masked_mean(
                crisis_probability, det_nonlow_mask
            ),
            "mean_share_exit_probability": masked_mean(
                exit_probability, det_nonlow_mask
            ),
        },
        "probability_region_fractions": {
            "p_low_at_least_0_50": float(np.mean(low_probability >= 0.50)),
            "p_low_at_least_0_75": float(np.mean(low_probability >= 0.75)),
            "p_crisis_at_least_0_50": float(np.mean(crisis_probability >= 0.50)),
            "p_exit_at_least_0_50": float(np.mean(exit_probability >= 0.50)),
        },
        "favourable_projection_nearest_cell": {
            "omega0": float(stochastic.omega_grid[favourable_index[0]]),
            "ell0": float(stochastic.ell_grid[favourable_index[1]]),
            "low_debt_probability": float(low_probability[favourable_index]),
            "positive_crisis_probability": float(crisis_probability[favourable_index]),
            "share_exit_probability": float(exit_probability[favourable_index]),
        },
        "unfavourable_projection_nearest_cell": {
            "omega0": float(stochastic.omega_grid[unfavourable_index[0]]),
            "ell0": float(stochastic.ell_grid[unfavourable_index[1]]),
            "low_debt_probability": float(low_probability[unfavourable_index]),
            "positive_crisis_probability": float(crisis_probability[unfavourable_index]),
            "share_exit_probability": float(exit_probability[unfavourable_index]),
        },
    }
    return summary


def export_cell_data(
    deterministic: DeterministicResult,
    stochastic: StochasticResult,
) -> Path:
    path = OUTPUT_DIRECTORY / CELL_DATA_NAME
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "omega0",
                "lambda0",
                "ell0",
                "deterministic_class_code",
                "deterministic_class",
                "stochastic_modal_class_code",
                "stochastic_modal_class",
                "stochastic_modal_probability",
                *[f"p_{CLASS_SHORT_NAMES[code]}" for code in CLASS_CODES],
            ]
        )

        for i, omega0 in enumerate(stochastic.omega_grid):
            for j, ell0 in enumerate(stochastic.ell_grid):
                det_code = int(deterministic.classes[i, j])
                modal_code = int(stochastic.modal_class[i, j])
                writer.writerow(
                    [
                        float(omega0),
                        LAMBDA_0,
                        float(ell0),
                        det_code,
                        CLASS_SHORT_NAMES[det_code],
                        modal_code,
                        CLASS_SHORT_NAMES[modal_code],
                        float(stochastic.modal_probability[i, j]),
                        *[
                            float(
                                stochastic.class_probabilities[
                                    i, j, CLASS_CODES.index(code)
                                ]
                            )
                            for code in CLASS_CODES
                        ],
                    ]
                )
    return path


def write_report(summary: Mapping[str, object]) -> Path:
    path = OUTPUT_DIRECTORY / SUMMARY_REPORT_NAME
    settings = summary["settings"]  # type: ignore[index]
    stochastic_means = summary["stochastic_grid_means"]  # type: ignore[index]
    inside = summary["inside_deterministic_low_debt_region"]  # type: ignore[index]
    outside = summary["outside_deterministic_low_debt_region"]  # type: ignore[index]
    regions = summary["probability_region_fractions"]  # type: ignore[index]

    lines = [
        "Deterministic classification and stochastic probability comparison",
        "=" * 78,
        f"lambda0: {settings['lambda0']:.8f}",  # type: ignore[index]
        f"grid: {settings['omega_points']} x {settings['ell_points']}",  # type: ignore[index]
        f"horizon: {settings['horizon']:.2f} years",  # type: ignore[index]
        f"stochastic dt: {settings['stochastic_dt']:.4f} years",  # type: ignore[index]
        f"stochastic paths per cell: {settings['paths_per_cell']}",  # type: ignore[index]
        "",
        "Whole-grid stochastic probability means",
        "-" * 78,
        f"Low-debt outcome: {100.0 * stochastic_means['low_debt_probability']:.2f}%",  # type: ignore[index]
        f"Positive-debt crisis: {100.0 * stochastic_means['positive_debt_crisis_probability']:.2f}%",  # type: ignore[index]
        f"Share-space exit: {100.0 * stochastic_means['share_space_exit_probability']:.2f}%",  # type: ignore[index]
        f"Unresolved: {100.0 * stochastic_means['unresolved_probability']:.2f}%",  # type: ignore[index]
        f"Mean modal probability: {100.0 * stochastic_means['modal_probability']:.2f}%",  # type: ignore[index]
        "",
        "Inside the deterministic low-debt region",
        "-" * 78,
        f"Cells: {inside['cells']}",  # type: ignore[index]
        f"Mean P(low debt): {100.0 * inside['mean_low_debt_probability']:.2f}%",  # type: ignore[index]
        f"Mean P(positive crisis): {100.0 * inside['mean_positive_crisis_probability']:.2f}%",  # type: ignore[index]
        f"Mean P(share exit): {100.0 * inside['mean_share_exit_probability']:.2f}%",  # type: ignore[index]
        f"Cells with P(low debt) >= 0.50: {100.0 * inside['fraction_cells_p_low_at_least_0_50']:.2f}%",  # type: ignore[index]
        f"Cells with P(low debt) >= 0.75: {100.0 * inside['fraction_cells_p_low_at_least_0_75']:.2f}%",  # type: ignore[index]
        "",
        "Outside the deterministic low-debt region",
        "-" * 78,
        f"Cells: {outside['cells']}",  # type: ignore[index]
        f"Mean P(low debt): {100.0 * outside['mean_low_debt_probability']:.2f}%",  # type: ignore[index]
        f"Mean P(positive crisis): {100.0 * outside['mean_positive_crisis_probability']:.2f}%",  # type: ignore[index]
        f"Mean P(share exit): {100.0 * outside['mean_share_exit_probability']:.2f}%",  # type: ignore[index]
        "",
        "Probability-region fractions over the full grid",
        "-" * 78,
        f"P(low debt) >= 0.50: {100.0 * regions['p_low_at_least_0_50']:.2f}%",  # type: ignore[index]
        f"P(low debt) >= 0.75: {100.0 * regions['p_low_at_least_0_75']:.2f}%",  # type: ignore[index]
        f"P(positive crisis) >= 0.50: {100.0 * regions['p_crisis_at_least_0_50']:.2f}%",  # type: ignore[index]
        f"P(share exit) >= 0.50: {100.0 * regions['p_exit_at_least_0_50']:.2f}%",  # type: ignore[index]
        "",
        "Interpretation note: stochastic probabilities are finite-horizon",
        "outcome frequencies under the stated numerical design, not basins of",
        "attraction or invariant-measure probabilities.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def export_outputs(
    deterministic: DeterministicResult,
    stochastic: StochasticResult,
) -> tuple[Path, Path, dict[str, object]]:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    summary = calculate_summary(deterministic, stochastic)

    cell_path = export_cell_data(deterministic, stochastic)
    json_path = OUTPUT_DIRECTORY / SUMMARY_JSON_NAME
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    report_path = write_report(summary)
    return cell_path, report_path, summary


# =============================================================================
# 10. Main workflow
# =============================================================================


def main(show: bool = True) -> None:
    validate_settings()
    parameters = load_baseline_parameters()

    total_cells = OMEGA_POINTS * ELL_POINTS
    total_paths = total_cells * N_STOCHASTIC_PATHS

    print("Single-slice deterministic/stochastic regime comparison")
    print("=" * 78)
    print(f"lambda0: {LAMBDA_0:.8f}")
    print(f"grid: {OMEGA_POINTS} x {ELL_POINTS} = {total_cells} cells")
    print(f"horizon: {T_MAP:.2f} years")
    print(f"stochastic dt: {DT_STOCHASTIC:.4f} years")
    print(f"stochastic paths per cell: {N_STOCHASTIC_PATHS}")
    print(f"total stochastic paths: {total_paths}")
    print(
        "baseline volatilities: "
        f"({SIGMA[0]:.3f}, {SIGMA[1]:.3f}, {SIGMA[2]:.3f})"
    )

    deterministic = simulate_deterministic_map(parameters)
    stochastic = simulate_stochastic_map(parameters)

    main_figure_path = create_main_figure(deterministic, stochastic)
    modal_figure_path = create_modal_diagnostic_figure(
        deterministic, stochastic
    )
    cell_path, report_path, summary = export_outputs(
        deterministic, stochastic
    )

    print("\n" + "=" * 78)
    print("Run complete")
    print("=" * 78)
    print(f"Main probability figure: {main_figure_path}")
    print(f"Modal diagnostic figure: {modal_figure_path}")
    print(f"Cell probabilities: {cell_path}")
    print(f"JSON summary: {OUTPUT_DIRECTORY / SUMMARY_JSON_NAME}")
    print(f"Text report: {report_path}")

    means = summary["stochastic_grid_means"]  # type: ignore[index]
    inside = summary["inside_deterministic_low_debt_region"]  # type: ignore[index]
    print(
        "Whole-grid mean P(low debt): "
        f"{100.0 * means['low_debt_probability']:.2f}%"  # type: ignore[index]
    )
    print(
        "Inside deterministic low-debt region, mean P(low debt): "
        f"{100.0 * inside['mean_low_debt_probability']:.2f}%"  # type: ignore[index]
    )

    if show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main(show=True)
