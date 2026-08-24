# deterministic_regime_maps.py
#
# Finite-horizon deterministic regime maps for the dividend-inclusive
# three-dimensional Keen model.
#
# Main methodological features:
#   1. Uses the calibrated low- and high-debt equilibria.
#   2. Integrates all deterministic paths with DOP853.
#   3. Distinguishes:
#        - convergence to the low-debt equilibrium,
#        - convergence to the high-debt equilibrium (if observed),
#        - share-space exit,
#        - positive-debt crisis: ell >= 50,
#        - negative-debt numerical safety cutoff,
#        - unresolved finite-horizon outcomes.
#   4. Computes the calibrated Jacobian eigenvalues at both equilibria.
#   5. Saves map summaries and numerical methodology to disk.
#
# The positive-debt crisis threshold is deliberately separate from the
# high-debt equilibrium. The latter is located at ell_H^* = 11.8, whereas
# the crisis stopping threshold is ell = 50.

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.integrate import solve_ivp

from keen_model_functions import KeenModel


# ============================================================
# 1. Calibration and equilibria
# ============================================================

DETERMINISTIC_PARAMS = {
    "sigma_omega": 0.0,
    "sigma_lambda": 0.0,
    "sigma_ell": 0.0,
}

LOW_DEBT_EQ = np.array(
    [0.6276666667, 0.6724861407, 0.0166666667],
    dtype=float,
)

HIGH_DEBT_EQ = np.array(
    [0.3920000000, 0.6543965885, 11.8000000000],
    dtype=float,
)

# ============================================================
# 2. Numerical methodology
# ============================================================

MAP_HORIZON = 500.0

SOLVER_METHOD = "DOP853"
RELATIVE_TOLERANCE = 1e-8
ABSOLUTE_TOLERANCE = 1e-10
MAX_INTERNAL_STEP = 0.50

# This is the precise "large positive debt" criterion used in the map.
POSITIVE_DEBT_CRISIS_THRESHOLD = 50.0

# A large negative value is a numerical safety cutoff only. It is not
# classified as a positive-debt crisis because ell < 0 represents net saving.
NEGATIVE_DEBT_SAFETY_FLOOR = -50.0

TAIL_DURATION = 100.0
TAIL_SAMPLES = 201

LOW_DISTANCE_SCALE = np.array([1.0, 1.0, 3.0], dtype=float)
HIGH_DISTANCE_SCALE = np.array([1.0, 1.0, 6.0], dtype=float)

LOW_FINAL_TOLERANCE = 0.06
LOW_TAIL_MEAN_TOLERANCE = 0.08
LOW_TAIL_STD_TOLERANCE = 0.03

HIGH_FINAL_TOLERANCE = 0.08
HIGH_TAIL_MEAN_TOLERANCE = 0.12
HIGH_TAIL_STD_TOLERANCE = 0.04

CACHE_VERSION = "regime_maps_v3"

OUTPUT_DIRECTORY = Path("Figures")
DATA_DIRECTORY = Path("NumericalData")
CACHE_DIRECTORY = DATA_DIRECTORY / "RegimeMapCache"


# ============================================================
# 3. Classification codes
# ============================================================

UNRESOLVED = -1
NEGATIVE_DEBT_CUTOFF = -2
POSITIVE_DEBT_CRISIS = 0
LOW_DEBT_CONVERGENCE = 1
HIGH_DEBT_CONVERGENCE = 2
SHARE_EXIT = 3

CLASS_LABELS = {
    UNRESOLVED: "Unresolved finite-horizon outcome",
    NEGATIVE_DEBT_CUTOFF: "Negative-debt safety cutoff",
    POSITIVE_DEBT_CRISIS: (
        rf"Positive-debt crisis: $\ell\geq"
        rf"{POSITIVE_DEBT_CRISIS_THRESHOLD:g}$"
    ),
    LOW_DEBT_CONVERGENCE: "Low-debt convergence",
    HIGH_DEBT_CONVERGENCE: "High-debt convergence",
    SHARE_EXIT: "Share-space exit",
}


# ============================================================
# 4. Model and calibrated Jacobian
# ============================================================

def make_deterministic_model() -> KeenModel:
    """Construct the deterministic calibrated model."""
    return KeenModel(DETERMINISTIC_PARAMS)


def local_truncated_slope(
    raw_value: float,
    lower: float,
    upper: float,
    interior_slope: float,
) -> float:
    """Return the local derivative of a truncated-linear function."""
    return interior_slope if lower < raw_value < upper else 0.0


def analytic_jacobian(
    model: KeenModel,
    state: Sequence[float],
) -> np.ndarray:
    """Evaluate the deterministic Jacobian derived in Section 4."""
    omega, lam, ell = map(float, state)
    p = model.params

    pi = float(model.profit_share(omega, ell))
    kap = float(model.kappa(pi))
    infl = float(model.inflation(omega))
    phi_value = float(model.phi(lam))

    raw_kappa = p["kappa0"] + p["kappa1"] * pi
    kappa_prime = local_truncated_slope(
        raw_kappa,
        p["kappa_min"],
        p["kappa_max"],
        p["kappa1"],
    )

    raw_dividend = p["dividend0"] + p["dividend1"] * pi
    dividend_prime = local_truncated_slope(
        raw_dividend,
        p["dividend_min"],
        p["dividend_max"],
        p["dividend1"],
    )

    inflation_prime = p["eta_p"] * p["m"]
    phillips_prime = p["phi1"]

    j_11 = (
        phi_value
        - p["alpha"]
        - (1.0 - p["gamma"]) * infl
        - omega * (1.0 - p["gamma"]) * inflation_prime
    )
    j_12 = omega * phillips_prime

    j_21 = -lam * kappa_prime / p["nu"]
    j_22 = (
        kap / p["nu"]
        - p["delta"]
        - p["alpha"]
        - p["beta"]
    )
    j_23 = -p["r"] * lam * kappa_prime / p["nu"]

    j_31 = (
        1.0
        + ell * kappa_prime / p["nu"]
        - ell * inflation_prime
        - kappa_prime
        - dividend_prime
    )
    j_33 = (
        p["r"]
        - kap / p["nu"]
        + p["delta"]
        - infl
        + p["r"]
        * (
            ell * kappa_prime / p["nu"]
            - kappa_prime
            - dividend_prime
        )
    )

    return np.array(
        [
            [j_11, j_12, 0.0],
            [j_21, j_22, j_23],
            [j_31, 0.0, j_33],
        ],
        dtype=float,
    )


def equilibrium_stability_report(
    model: KeenModel,
) -> dict[str, object]:
    """Compute Jacobians, eigenvalues, and local stability classifications."""
    report: dict[str, object] = {}

    for name, equilibrium in (
        ("low_debt", LOW_DEBT_EQ),
        ("high_debt", HIGH_DEBT_EQ),
    ):
        jacobian = analytic_jacobian(model, equilibrium)
        eigenvalues = np.linalg.eigvals(jacobian)
        maximum_real_part = float(np.max(eigenvalues.real))

        if maximum_real_part < 0.0:
            classification = "locally asymptotically stable"
        elif maximum_real_part > 0.0:
            classification = "locally unstable"
        else:
            classification = "non-hyperbolic or numerically inconclusive"

        report[name] = {
            "equilibrium": equilibrium.tolist(),
            "jacobian": jacobian.tolist(),
            "eigenvalues": [
                {
                    "real": float(value.real),
                    "imag": float(value.imag),
                }
                for value in eigenvalues
            ],
            "maximum_real_part": maximum_real_part,
            "classification": classification,
        }

    return report


# ============================================================
# 5. Deterministic integration and stopping events
# ============================================================

def make_events():
    """Create labelled terminal events for the deterministic integration."""
    event_names = [
        "omega_lower_exit",
        "omega_upper_exit",
        "lambda_lower_exit",
        "lambda_upper_exit",
        "positive_debt_threshold",
        "negative_debt_safety_cutoff",
    ]

    def omega_lower_event(time, state):
        del time
        return state[0]

    omega_lower_event.terminal = True
    omega_lower_event.direction = -1

    def omega_upper_event(time, state):
        del time
        return 1.0 - state[0]

    omega_upper_event.terminal = True
    omega_upper_event.direction = -1

    def lambda_lower_event(time, state):
        del time
        return state[1]

    lambda_lower_event.terminal = True
    lambda_lower_event.direction = -1

    def lambda_upper_event(time, state):
        del time
        return 1.0 - state[1]

    lambda_upper_event.terminal = True
    lambda_upper_event.direction = -1

    def positive_debt_event(time, state):
        del time
        return POSITIVE_DEBT_CRISIS_THRESHOLD - state[2]

    positive_debt_event.terminal = True
    positive_debt_event.direction = -1

    def negative_debt_event(time, state):
        del time
        return state[2] - NEGATIVE_DEBT_SAFETY_FLOOR

    negative_debt_event.terminal = True
    negative_debt_event.direction = -1

    events = [
        omega_lower_event,
        omega_upper_event,
        lambda_lower_event,
        lambda_upper_event,
        positive_debt_event,
        negative_debt_event,
    ]

    return event_names, events


def identify_stop_reason(
    solution,
    event_names: list[str],
) -> str:
    """Identify the event that stopped a solve_ivp integration."""
    if solution.status != 1:
        return "completed"

    event_hits = [
        (event_times[0], event_names[index])
        for index, event_times in enumerate(solution.t_events)
        if len(event_times) > 0
    ]

    if not event_hits:
        return "event_without_label"

    return min(event_hits, key=lambda item: item[0])[1]


def integrate_deterministic(
    model: KeenModel,
    omega0: float,
    lambda0: float,
    ell0: float,
    *,
    horizon: float = MAP_HORIZON,
):
    """Integrate one deterministic path with explicit stopping criteria."""
    event_names, events = make_events()

    solution = solve_ivp(
        fun=model.deterministic_rhs,
        t_span=(0.0, horizon),
        y0=np.array([omega0, lambda0, ell0], dtype=float),
        method=SOLVER_METHOD,
        rtol=RELATIVE_TOLERANCE,
        atol=ABSOLUTE_TOLERANCE,
        max_step=MAX_INTERNAL_STEP,
        events=events,
        dense_output=True,
    )

    if not solution.success:
        raise RuntimeError(
            "Deterministic integration failed: " + solution.message
        )

    stop_reason = identify_stop_reason(solution, event_names)
    final_state = solution.y[:, -1].astype(float)

    return solution, final_state, stop_reason


# ============================================================
# 6. Convergence diagnostics and classification
# ============================================================

def scaled_distances(
    states: np.ndarray,
    equilibrium: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    """Return dimensionless scaled Euclidean distances."""
    normalized = (states - equilibrium) / scales
    return np.sqrt(np.sum(normalized**2, axis=1))


def tail_distance_statistics(
    solution,
    equilibrium: np.ndarray,
    scales: np.ndarray,
    *,
    horizon: float = MAP_HORIZON,
) -> tuple[float, float, float]:
    """Calculate final and time-uniform tail distance statistics."""
    tail_start = max(0.0, horizon - TAIL_DURATION)
    tail_times = np.linspace(
        tail_start,
        horizon,
        TAIL_SAMPLES,
        dtype=float,
    )
    tail_states = solution.sol(tail_times).T

    tail_distances = scaled_distances(
        tail_states,
        equilibrium,
        scales,
    )
    final_distance = float(tail_distances[-1])

    return (
        final_distance,
        float(np.mean(tail_distances)),
        float(np.std(tail_distances)),
    )


def classify_initial_condition(
    model: KeenModel,
    omega0: float,
    lambda0: float,
    ell0: float,
    *,
    horizon: float = MAP_HORIZON,
) -> tuple[int, dict[str, object]]:
    """Classify one finite-horizon deterministic trajectory."""
    solution, final_state, stop_reason = integrate_deterministic(
        model,
        omega0,
        lambda0,
        ell0,
        horizon=horizon,
    )

    diagnostics: dict[str, object] = {
        "stop_reason": stop_reason,
        "stop_time": float(solution.t[-1]),
        "omega_final": float(final_state[0]),
        "lambda_final": float(final_state[1]),
        "ell_final": float(final_state[2]),
    }

    if stop_reason in {
        "omega_lower_exit",
        "omega_upper_exit",
        "lambda_lower_exit",
        "lambda_upper_exit",
    }:
        return SHARE_EXIT, diagnostics

    if stop_reason == "positive_debt_threshold":
        return POSITIVE_DEBT_CRISIS, diagnostics

    if stop_reason == "negative_debt_safety_cutoff":
        return NEGATIVE_DEBT_CUTOFF, diagnostics

    if stop_reason != "completed":
        return UNRESOLVED, diagnostics

    low_stats = tail_distance_statistics(
        solution,
        LOW_DEBT_EQ,
        LOW_DISTANCE_SCALE,
        horizon=horizon,
    )
    high_stats = tail_distance_statistics(
        solution,
        HIGH_DEBT_EQ,
        HIGH_DISTANCE_SCALE,
        horizon=horizon,
    )

    diagnostics.update(
        {
            "low_final_distance": low_stats[0],
            "low_tail_mean_distance": low_stats[1],
            "low_tail_std_distance": low_stats[2],
            "high_final_distance": high_stats[0],
            "high_tail_mean_distance": high_stats[1],
            "high_tail_std_distance": high_stats[2],
        }
    )

    low_condition = (
        low_stats[0] < LOW_FINAL_TOLERANCE
        and low_stats[1] < LOW_TAIL_MEAN_TOLERANCE
        and low_stats[2] < LOW_TAIL_STD_TOLERANCE
    )
    high_condition = (
        high_stats[0] < HIGH_FINAL_TOLERANCE
        and high_stats[1] < HIGH_TAIL_MEAN_TOLERANCE
        and high_stats[2] < HIGH_TAIL_STD_TOLERANCE
    )

    if low_condition and not high_condition:
        return LOW_DEBT_CONVERGENCE, diagnostics

    if high_condition and not low_condition:
        return HIGH_DEBT_CONVERGENCE, diagnostics

    if low_condition and high_condition:
        if low_stats[1] <= high_stats[1]:
            return LOW_DEBT_CONVERGENCE, diagnostics
        return HIGH_DEBT_CONVERGENCE, diagnostics

    return UNRESOLVED, diagnostics


# ============================================================
# 7. Grid computation and caching
# ============================================================

def cache_path_for_slice(
    lambda0: float,
    n_omega: int,
    n_ell: int,
    horizon: float,
) -> Path:
    """Return a versioned cache path for one regime-map slice."""
    safe_lambda = f"{lambda0:.7f}".replace(".", "p")
    safe_horizon = f"{horizon:g}".replace(".", "p")

    filename = (
        f"{CACHE_VERSION}_lambda_{safe_lambda}"
        f"_omega{n_omega}_ell{n_ell}"
        f"_T{safe_horizon}"
        f"_debt{POSITIVE_DEBT_CRISIS_THRESHOLD:g}.npz"
    )
    return CACHE_DIRECTORY / filename


def summarize_regime(
    lambda0: float,
    regime: np.ndarray,
) -> dict[str, int | float]:
    """Count each classification in one regime grid."""
    return {
        "lambda0": float(lambda0),
        "low_debt_convergence": int(
            np.sum(regime == LOW_DEBT_CONVERGENCE)
        ),
        "high_debt_convergence": int(
            np.sum(regime == HIGH_DEBT_CONVERGENCE)
        ),
        "positive_debt_crisis": int(
            np.sum(regime == POSITIVE_DEBT_CRISIS)
        ),
        "share_exit": int(np.sum(regime == SHARE_EXIT)),
        "negative_debt_cutoff": int(
            np.sum(regime == NEGATIVE_DEBT_CUTOFF)
        ),
        "unresolved": int(np.sum(regime == UNRESOLVED)),
        "total": int(regime.size),
    }


def compute_regime_grid(
    lambda0: float,
    *,
    omega_min: float = 0.20,
    omega_max: float = 0.95,
    ell_min: float = -1.0,
    ell_max: float = 14.0,
    n_omega: int = 101,
    n_ell: int = 121,
    horizon: float = MAP_HORIZON,
    verbose: bool = True,
    use_cache: bool = True,
):
    """Compute or load one fixed-employment regime-map slice."""
    CACHE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    cache_path = cache_path_for_slice(
        lambda0,
        n_omega,
        n_ell,
        horizon,
    )

    if use_cache and cache_path.exists():
        cached = np.load(cache_path)
        omega_grid = cached["omega_grid"]
        ell_grid = cached["ell_grid"]
        regime = cached["regime"]
        summary = summarize_regime(lambda0, regime)

        if verbose:
            print(f"Loaded cached slice: {cache_path}")

        return omega_grid, ell_grid, regime, summary

    model = make_deterministic_model()
    omega_grid = np.linspace(
        omega_min,
        omega_max,
        n_omega,
    )
    ell_grid = np.linspace(
        ell_min,
        ell_max,
        n_ell,
    )

    regime = np.full(
        (n_omega, n_ell),
        UNRESOLVED,
        dtype=np.int8,
    )

    for row, omega0 in enumerate(omega_grid):
        if verbose and row % 10 == 0:
            print(
                f"lambda0={lambda0:.4f}: "
                f"row {row + 1}/{n_omega}"
            )

        for column, ell0 in enumerate(ell_grid):
            code, _ = classify_initial_condition(
                model,
                float(omega0),
                float(lambda0),
                float(ell0),
                horizon=horizon,
            )
            regime[row, column] = code

    np.savez_compressed(
        cache_path,
        omega_grid=omega_grid,
        ell_grid=ell_grid,
        regime=regime,
    )

    summary = summarize_regime(lambda0, regime)

    return omega_grid, ell_grid, regime, summary


# ============================================================
# 8. Plotting
# ============================================================

PLOT_ORDER = [
    UNRESOLVED,
    NEGATIVE_DEBT_CUTOFF,
    SHARE_EXIT,
    POSITIVE_DEBT_CRISIS,
    LOW_DEBT_CONVERGENCE,
    HIGH_DEBT_CONVERGENCE,
]

PLOT_COLORS = [
    "#d9d9d9",  # unresolved
    "#969696",  # negative-debt safety cutoff
    "#D2F9DB",  # share-space exit
    "#e57373",  # positive-debt crisis
    "#63E981",  # low-debt convergence
    "#f6c16b",  # high-debt convergence
]


def regime_to_plot_index(
    regime: np.ndarray,
) -> np.ndarray:
    """Convert classification codes to consecutive plotting indices."""
    plot_index = np.zeros_like(regime, dtype=int)

    for index, code in enumerate(PLOT_ORDER):
        plot_index[regime == code] = index

    return plot_index


def add_markers(
    ax: plt.Axes,
    lambda0: float,
    ell_min: float,
    ell_max: float,
    omega_min: float,
    omega_max: float,
) -> None:
    """Add equilibrium and selected scenario projections."""
    omega_low, _, ell_low = LOW_DEBT_EQ
    omega_high, _, ell_high = HIGH_DEBT_EQ

    if ell_min <= ell_low <= ell_max and omega_min <= omega_low <= omega_max:
        ax.plot(
            ell_low,
            omega_low,
            marker="*",
            color="black",
            markersize=14,
            linestyle="None",
            zorder=6,
        )

    if (
        ell_min <= ell_high <= ell_max
        and omega_min <= omega_high <= omega_max
    ):
        ax.plot(
            ell_high,
            omega_high,
            marker="D",
            color="black",
            markersize=8,
            linestyle="None",
            zorder=6,
        )

    # Favorable and unfavorable calibration points.
    reference_points = [
        (0.30, 0.90, "o"),
        (1.53, 0.578, "s"),
    ]
    for ell_value, omega_value, marker in reference_points:
        if (
            ell_min <= ell_value <= ell_max
            and omega_min <= omega_value <= omega_max
        ):
            ax.plot(
                ell_value,
                omega_value,
                marker=marker,
                markerfacecolor="white",
                markeredgecolor="black",
                markersize=6,
                linestyle="None",
                zorder=7,
            )


def plot_single_regime_map(
    ax: plt.Axes,
    omega_grid: np.ndarray,
    ell_grid: np.ndarray,
    regime: np.ndarray,
    lambda0: float,
) -> None:
    """Plot one fixed-employment finite-horizon regime map."""
    debt_mesh, wage_mesh = np.meshgrid(
        ell_grid,
        omega_grid,
    )
    plot_index = regime_to_plot_index(regime)

    cmap = ListedColormap(PLOT_COLORS)
    norm = BoundaryNorm(
        np.arange(-0.5, len(PLOT_ORDER) + 0.5, 1.0),
        cmap.N,
    )

    ax.pcolormesh(
        debt_mesh,
        wage_mesh,
        plot_index,
        cmap=cmap,
        norm=norm,
        shading="auto",
        alpha=0.88,
    )

    low_mask = (
        regime == LOW_DEBT_CONVERGENCE
    ).astype(float)

    if np.any(low_mask > 0.0):
        ax.contour(
            debt_mesh,
            wage_mesh,
            low_mask,
            levels=[0.5],
            colors="#1f4aff",
            linewidths=1.5,
        )

    ell_min = float(ell_grid.min())
    ell_max = float(ell_grid.max())
    omega_min = float(omega_grid.min())
    omega_max = float(omega_grid.max())

    add_markers(
        ax,
        lambda0,
        ell_min,
        ell_max,
        omega_min,
        omega_max,
    )

    ax.set_title(rf"$\lambda_0={lambda0:.4f}$")
    ax.set_xlabel(r"Initial debt $\ell_0$")
    ax.set_ylabel(r"Initial wage share $\omega_0$")
    ax.set_xlim(ell_min, ell_max)
    ax.set_ylim(omega_min, omega_max)
    ax.grid(True, alpha=0.25)


def legend_elements() -> list:
    """Return the common figure legend."""
    return [
        Patch(
            facecolor=PLOT_COLORS[4],
            edgecolor="none",
            alpha=0.88,
            label="Low-debt convergence",
        ),
        Patch(
            facecolor=PLOT_COLORS[5],
            edgecolor="none",
            alpha=0.88,
            label="High-debt convergence (if observed)",
        ),
        Patch(
            facecolor=PLOT_COLORS[3],
            edgecolor="none",
            alpha=0.88,
            label=(
                rf"Positive-debt crisis "
                rf"$(\ell\geq"
                rf"{POSITIVE_DEBT_CRISIS_THRESHOLD:g})$"
            ),
        ),
        Patch(
            facecolor=PLOT_COLORS[2],
            edgecolor="none",
            alpha=0.88,
            label="Share-space exit",
        ),
        Patch(
            facecolor=PLOT_COLORS[0],
            edgecolor="none",
            alpha=0.88,
            label="Unresolved",
        ),
        Patch(
            facecolor=PLOT_COLORS[1],
            edgecolor="none",
            alpha=0.88,
            label="Negative-debt safety cutoff",
        ),
        Line2D(
            [0],
            [0],
            color="#1f4aff",
            linewidth=1.8,
            label="Boundary of low-debt region",
        ),
        Line2D(
            [0],
            [0],
            marker="*",
            color="black",
            markersize=12,
            linestyle="None",
            label="Low-debt equilibrium projection",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            color="black",
            markersize=8,
            linestyle="None",
            label="High-debt equilibrium projection",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            markerfacecolor="white",
            markeredgecolor="black",
            color="none",
            markersize=6,
            linestyle="None",
            label="Favourable initial-condition projection",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            markerfacecolor="white",
            markeredgecolor="black",
            color="none",
            markersize=6,
            linestyle="None",
            label="Unfavourable initial-condition projection",
        ),
    ]


# ============================================================
# 9. Output helpers
# ============================================================

def save_methodology(
    stability: dict[str, object],
    summaries: list[dict[str, object]],
) -> tuple[Path, Path]:
    """Save methodology, stability, and map-summary outputs."""
    DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)

    methodology_path = (
        DATA_DIRECTORY
        / "deterministic_regime_map_methodology.json"
    )
    summary_path = (
        DATA_DIRECTORY
        / "deterministic_regime_map_summaries.csv"
    )

    payload = {
        "solver": {
            "method": SOLVER_METHOD,
            "relative_tolerance": RELATIVE_TOLERANCE,
            "absolute_tolerance": ABSOLUTE_TOLERANCE,
            "maximum_internal_step": MAX_INTERNAL_STEP,
        },
        "classification": {
            "horizon": MAP_HORIZON,
            "positive_debt_crisis_threshold": (
                POSITIVE_DEBT_CRISIS_THRESHOLD
            ),
            "high_debt_equilibrium_debt": float(HIGH_DEBT_EQ[2]),
            "threshold_to_high_equilibrium_ratio": float(
                POSITIVE_DEBT_CRISIS_THRESHOLD / HIGH_DEBT_EQ[2]
            ),
            "negative_debt_safety_floor": (
                NEGATIVE_DEBT_SAFETY_FLOOR
            ),
            "tail_duration": TAIL_DURATION,
            "tail_samples": TAIL_SAMPLES,
            "low_distance_scale": LOW_DISTANCE_SCALE.tolist(),
            "high_distance_scale": HIGH_DISTANCE_SCALE.tolist(),
            "low_tolerances": {
                "final": LOW_FINAL_TOLERANCE,
                "tail_mean": LOW_TAIL_MEAN_TOLERANCE,
                "tail_std": LOW_TAIL_STD_TOLERANCE,
            },
            "high_tolerances": {
                "final": HIGH_FINAL_TOLERANCE,
                "tail_mean": HIGH_TAIL_MEAN_TOLERANCE,
                "tail_std": HIGH_TAIL_STD_TOLERANCE,
            },
        },
        "equilibrium_stability": stability,
        "slice_summaries": summaries,
    }

    with methodology_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)

    columns = [
        "lambda0",
        "low_debt_convergence",
        "high_debt_convergence",
        "positive_debt_crisis",
        "share_exit",
        "negative_debt_cutoff",
        "unresolved",
        "total",
    ]

    table = np.array(
        [
            [summary[column] for column in columns]
            for summary in summaries
        ],
        dtype=float,
    )

    np.savetxt(
        summary_path,
        table,
        delimiter=",",
        header=",".join(columns),
        comments="",
    )

    return methodology_path, summary_path


def print_stability_report(
    stability: dict[str, object],
) -> None:
    """Print the calibrated eigenvalues at both equilibria."""
    print("\nCalibrated equilibrium stability")
    print("=" * 64)

    for key, title in (
        ("low_debt", "Low-debt equilibrium"),
        ("high_debt", "High-debt equilibrium"),
    ):
        item = stability[key]
        print(f"\n{title}")
        print("-" * 64)

        for eigenvalue in item["eigenvalues"]:
            real = eigenvalue["real"]
            imag = eigenvalue["imag"]

            if abs(imag) < 1e-12:
                print(f"{real:.8f}")
            else:
                sign = "+" if imag >= 0 else "-"
                print(
                    f"{real:.8f}"
                    f"{sign}"
                    f"{abs(imag):.8f}i"
                )

        print(
            "Maximum real part: "
            f"{item['maximum_real_part']:.10f}"
        )
        print(f"Classification: {item['classification']}")


# ============================================================
# 10. Figure creators
# ============================================================

def create_multi_lambda_regime_maps(
    *,
    lambda_slices: Sequence[float] | None = None,
    omega_min: float = 0.20,
    omega_max: float = 0.95,
    ell_min: float = -1.0,
    ell_max: float = 14.0,
    n_omega: int = 101,
    n_ell: int = 121,
    horizon: float = MAP_HORIZON,
    use_cache: bool = True,
):
    """Create the four-slice finite-horizon regime-map figure."""
    if lambda_slices is None:
        lambda_slices = [
            0.50,
            float(LOW_DEBT_EQ[1]),
            0.75,
            0.95,
        ]

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(13.5, 10.0),
        sharex=True,
        sharey=True,
    )
    axes = axes.ravel()

    summaries: list[dict[str, object]] = []

    for axis, lambda0 in zip(axes, lambda_slices):
        print(f"\nComputing regime map for lambda0={lambda0:.4f}")

        omega_grid, ell_grid, regime, summary = compute_regime_grid(
            float(lambda0),
            omega_min=omega_min,
            omega_max=omega_max,
            ell_min=ell_min,
            ell_max=ell_max,
            n_omega=n_omega,
            n_ell=n_ell,
            horizon=horizon,
            verbose=True,
            use_cache=use_cache,
        )

        summaries.append(summary)
        plot_single_regime_map(
            axis,
            omega_grid,
            ell_grid,
            regime,
            float(lambda0),
        )

        print(json.dumps(summary, indent=2))

    figure.suptitle(
        "Finite-Horizon Deterministic Regime Maps",
        fontsize=16,
        fontweight="bold",
        y=0.97,
    )

    figure.legend(
        handles=legend_elements(),
        loc="lower center",
        ncol=3,
        frameon=True,
        bbox_to_anchor=(0.5, 0.005),
        fontsize=9,
    )

    figure.subplots_adjust(
        left=0.07,
        right=0.98,
        top=0.91,
        bottom=0.20,
        hspace=0.22,
        wspace=0.14,
    )

    output_path = (
        OUTPUT_DIRECTORY
        / "deterministic_regime_maps_multi_lambda.png"
    )
    pdf_path = output_path.with_suffix(".pdf")

    figure.savefig(output_path, dpi=300)
    figure.savefig(pdf_path)

    print(f"\nSaved multi-slice map: {output_path}")

    return figure, summaries


def create_single_lambda_regime_map(
    *,
    lambda0: float = float(LOW_DEBT_EQ[1]),
    omega_min: float = 0.20,
    omega_max: float = 0.95,
    ell_min: float = -1.0,
    ell_max: float = 14.0,
    n_omega: int = 121,
    n_ell: int = 141,
    horizon: float = MAP_HORIZON,
    use_cache: bool = True,
):
    """Create the detailed low-debt-equilibrium employment slice."""
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    omega_grid, ell_grid, regime, summary = compute_regime_grid(
        lambda0,
        omega_min=omega_min,
        omega_max=omega_max,
        ell_min=ell_min,
        ell_max=ell_max,
        n_omega=n_omega,
        n_ell=n_ell,
        horizon=horizon,
        verbose=True,
        use_cache=use_cache,
    )

    figure, axis = plt.subplots(figsize=(10.2, 7.4))

    plot_single_regime_map(
        axis,
        omega_grid,
        ell_grid,
        regime,
        lambda0,
    )

    axis.set_title(
        rf"Finite-Horizon Deterministic Regime Map "
        rf"$(\lambda_0={lambda0:.4f})$",
        fontsize=15,
        fontweight="bold",
    )
    axis.legend(
        handles=legend_elements(),
        loc="upper right",
        frameon=True,
        fontsize=8.5,
    )

    figure.subplots_adjust(
        left=0.10,
        right=0.98,
        top=0.91,
        bottom=0.10,
    )

    output_path = (
        OUTPUT_DIRECTORY
        / "deterministic_regime_map_lambda_low_equilibrium.png"
    )
    pdf_path = output_path.with_suffix(".pdf")

    figure.savefig(output_path, dpi=300)
    figure.savefig(pdf_path)

    print(f"\nSaved single-slice map: {output_path}")
    print(json.dumps(summary, indent=2))

    return figure, summary


# ============================================================
# 11. Main execution
# ============================================================

if __name__ == "__main__":
    print("Creating revised deterministic regime maps...")
    print(f"Low-debt equilibrium:  {LOW_DEBT_EQ}")
    print(f"High-debt equilibrium: {HIGH_DEBT_EQ}")
    print(
        "Positive-debt crisis threshold: "
        f"ell >= {POSITIVE_DEBT_CRISIS_THRESHOLD:g}"
    )
    print(
        "This threshold is "
        f"{POSITIVE_DEBT_CRISIS_THRESHOLD / HIGH_DEBT_EQ[2]:.3f} "
        "times the high-debt equilibrium debt coordinate."
    )

    deterministic_model = make_deterministic_model()
    stability_report = equilibrium_stability_report(
        deterministic_model
    )
    print_stability_report(stability_report)

    multi_figure, multi_summaries = (
        create_multi_lambda_regime_maps(
            lambda_slices=[
                0.50,
                float(LOW_DEBT_EQ[1]),
                0.75,
                0.95,
            ],
            omega_min=0.20,
            omega_max=0.95,
            ell_min=-1.0,
            ell_max=14.0,
            n_omega=101,
            n_ell=121,
            horizon=MAP_HORIZON,
            use_cache=True,
        )
    )

    single_figure, single_summary = (
        create_single_lambda_regime_map(
            lambda0=float(LOW_DEBT_EQ[1]),
            omega_min=0.20,
            omega_max=0.95,
            ell_min=-1.0,
            ell_max=14.0,
            n_omega=121,
            n_ell=141,
            horizon=MAP_HORIZON,
            use_cache=True,
        )
    )

    all_summaries = [
        *multi_summaries,
        {
            **single_summary,
            "figure": "single_low_equilibrium_slice",
        },
    ]

    methodology_path, summary_path = save_methodology(
        stability_report,
        all_summaries,
    )

    print(f"\nSaved methodology: {methodology_path}")
    print(f"Saved summaries:    {summary_path}")

    plt.show()
