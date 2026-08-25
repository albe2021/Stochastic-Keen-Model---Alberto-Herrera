# baseline_dynamics.py
#
# Creates the baseline stochastic figure:
#   (a) state-variable time series
#   (b) three-dimensional phase-space trajectory
#   (c) debt service, profit share, and inflation
#
# implemented in keen_model_functions.py.

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from keen_model_functions import KeenModel


# ============================================================
# Plot configuration
# ============================================================
plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)


# ============================================================
# Baseline calibration
# ============================================================
BASELINE_PARAMS = {
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

BASELINE_SEED = 128
BASELINE_INITIAL_CONDITION = (0.9, 0.9, 0.3)
T_SIM = 200.0
DT = 0.01

# Plotting intervals. Full-frequency data are retained in the CSV output.
TIME_SERIES_INTERVAL = 0.10
PHASE_INTERVAL = 0.25
PHASE_SMOOTHING_WINDOW = 1

OUTPUT_DIRECTORY = Path("Figures")
DATA_DIRECTORY = Path("NumericalData")
OUTPUT_STEM = "baseline_dynamics"


# ============================================================
# Shared thesis colour convention
# ============================================================
# State variables
COLOR_OMEGA = "#1f77b4"          # blue
COLOR_LAMBDA = "#d62728"         # red
COLOR_ELL = "#2ca02c"            # green

# Financial indicators
COLOR_PROFIT = "#1f77b4"         # blue
COLOR_DEBT_SERVICE = "#9467bd"   # purple
COLOR_INFLATION = "#ff7f0e"      # orange

# Plot annotations
COLOR_REFERENCE = "#666666"
COLOR_OUTSIDE_REGION = "#bdbdbd"
COLOR_DEFLATION_SHADE = "#f4a261"
COLOR_START = COLOR_OMEGA
COLOR_END = COLOR_INFLATION
PHASE_CMAP = "viridis"


# ============================================================
# Helper functions
# ============================================================
def contiguous_true_regions(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return contiguous True regions as half-open intervals [start, end)."""
    mask = np.asarray(mask, dtype=bool)

    if mask.ndim != 1:
        raise ValueError("mask must be one-dimensional.")
    if len(mask) == 0:
        return []

    changes = np.diff(mask.astype(np.int8))
    starts = np.flatnonzero(changes == 1) + 1
    ends = np.flatnonzero(changes == -1) + 1

    if mask[0]:
        starts = np.insert(starts, 0, 0)
    if mask[-1]:
        ends = np.append(ends, len(mask))

    return list(zip(starts.tolist(), ends.tolist()))


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    """Return a centred moving average with edge padding."""
    x = np.asarray(x, dtype=float)

    if window <= 1:
        return x.copy()
    if window > len(x):
        raise ValueError("The moving-average window exceeds the data length.")

    left = window // 2
    right = window - 1 - left
    padded = np.pad(x, (left, right), mode="edge")
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(padded, kernel, mode="valid")


def first_true_time(t: np.ndarray, mask: np.ndarray) -> float | None:
    """Return the first time at which mask is True, or None."""
    indices = np.flatnonzero(mask)
    return float(t[indices[0]]) if len(indices) else None


def longest_duration(mask: np.ndarray, dt: float) -> float:
    """Return the longest contiguous True duration."""
    regions = contiguous_true_regions(mask)
    if not regions:
        return 0.0
    return float(max((end - start) * dt for start, end in regions))


def sample_indices(
    t: np.ndarray,
    interval: float,
) -> np.ndarray:
    """Return approximately equally spaced indices for plotting."""
    if interval <= 0:
        raise ValueError("interval must be positive.")

    dt = float(t[1] - t[0])
    step = max(1, int(round(interval / dt)))
    indices = np.arange(0, len(t), step, dtype=int)

    if indices[-1] != len(t) - 1:
        indices = np.append(indices, len(t) - 1)

    return indices


def add_axis_padding(
    ax,
    values: np.ndarray,
    axis: str,
    fraction: float = 0.05,
) -> None:
    """Add modest data-driven padding to a 3D axis."""
    values = np.asarray(values, dtype=float)
    lower = float(np.min(values))
    upper = float(np.max(values))
    span = upper - lower
    padding = fraction * span if span > 0 else 0.05
    limits = (lower - padding, upper + padding)

    if axis == "x":
        ax.set_xlim(*limits)
    elif axis == "y":
        ax.set_ylim(*limits)
    elif axis == "z":
        ax.set_zlim(*limits)
    else:
        raise ValueError("axis must be 'x', 'y', or 'z'.")


# ============================================================
# Simulation and diagnostics
# ============================================================
def simulate_baseline_path(
    seed: int = BASELINE_SEED,
    T: float = T_SIM,
    dt: float = DT,
    x0: Sequence[float] = BASELINE_INITIAL_CONDITION,
) -> tuple[KeenModel, np.ndarray, np.ndarray]:
    """
    Simulate one unprojected baseline stochastic path.

    If omega or lambda lies outside [0, 1], its direct Brownian
    coefficient is zero while its drift remains active.
    """
    model = KeenModel(BASELINE_PARAMS)
    t, states = model.simulate_path(
        x0=x0,
        T=T,
        dt=dt,
        seed=seed,
    )

    if not np.all(np.isfinite(states)):
        raise FloatingPointError(
            "The baseline simulation produced non-finite states."
        )

    return model, t, states


def calculate_derived_variables(
    model: KeenModel,
    states: np.ndarray,
) -> dict[str, np.ndarray]:
    """Calculate financial ratios and boundary indicators."""
    omega = states[:, 0]
    lam = states[:, 1]
    ell = states[:, 2]

    profit = np.asarray(model.profit_share(omega, ell), dtype=float)
    debt_service = model.params["r"] * ell
    inflation = np.asarray(model.inflation(omega), dtype=float)
    financing_gap = np.asarray(
        model.financing_gap(omega, lam, ell),
        dtype=float,
    )
    interest_growth_gap = np.asarray(
        model.interest_growth_gap(omega, ell),
        dtype=float,
    )

    omega_below = omega < 0.0
    omega_above = omega > 1.0
    lambda_below = lam < 0.0
    lambda_above = lam > 1.0
    outside_economic_region = (
        omega_below
        | omega_above
        | lambda_below
        | lambda_above
    )

    return {
        "profit": profit,
        "debt_service": debt_service,
        "inflation": inflation,
        "financing_gap": financing_gap,
        "interest_growth_gap": interest_growth_gap,
        "omega_below_zero": omega_below,
        "omega_above_one": omega_above,
        "lambda_below_zero": lambda_below,
        "lambda_above_one": lambda_above,
        "outside_economic_region": outside_economic_region,
        "deflation": inflation < 0.0,
        "net_savings": ell < 0.0,
    }


def summarize_baseline_path(
    t: np.ndarray,
    states: np.ndarray,
    derived: dict[str, np.ndarray],
) -> dict[str, float | int | None]:
    """Calculate statistics used in the revised Section 5.2."""
    omega = states[:, 0]
    lam = states[:, 1]
    ell = states[:, 2]
    dt = float(t[1] - t[0])

    outside = derived["outside_economic_region"]
    omega_above = derived["omega_above_one"]
    omega_below = derived["omega_below_zero"]
    lambda_above = derived["lambda_above_one"]
    lambda_below = derived["lambda_below_zero"]

    return {
        "seed": BASELINE_SEED,
        "T": float(t[-1]),
        "dt": dt,
        "omega_min": float(np.min(omega)),
        "omega_max": float(np.max(omega)),
        "lambda_min": float(np.min(lam)),
        "lambda_max": float(np.max(lam)),
        "ell_min": float(np.min(ell)),
        "ell_max": float(np.max(ell)),
        "ell_mean": float(np.mean(ell)),
        "ell_std": float(np.std(ell)),
        "profit_min": float(np.min(derived["profit"])),
        "profit_max": float(np.max(derived["profit"])),
        "debt_service_min": float(np.min(derived["debt_service"])),
        "debt_service_max": float(np.max(derived["debt_service"])),
        "inflation_min": float(np.min(derived["inflation"])),
        "inflation_max": float(np.max(derived["inflation"])),
        "financing_gap_min": float(np.min(derived["financing_gap"])),
        "financing_gap_max": float(np.max(derived["financing_gap"])),
        "net_savings_fraction": float(np.mean(derived["net_savings"])),
        "deflation_fraction": float(np.mean(derived["deflation"])),
        "longest_deflation_duration": longest_duration(
            derived["deflation"],
            dt,
        ),
        "economic_region_fraction": float(np.mean(~outside)),
        "outside_economic_region_fraction": float(np.mean(outside)),
        "first_exit_time": first_true_time(t, outside),
        "omega_first_above_one": first_true_time(t, omega_above),
        "omega_first_below_zero": first_true_time(t, omega_below),
        "lambda_first_above_one": first_true_time(t, lambda_above),
        "lambda_first_below_zero": first_true_time(t, lambda_below),
        "omega_above_one_fraction": float(np.mean(omega_above)),
        "omega_below_zero_fraction": float(np.mean(omega_below)),
        "lambda_above_one_fraction": float(np.mean(lambda_above)),
        "lambda_below_zero_fraction": float(np.mean(lambda_below)),
        "omega_above_one_episodes": len(
            contiguous_true_regions(omega_above)
        ),
        "lambda_above_one_episodes": len(
            contiguous_true_regions(lambda_above)
        ),
        "longest_omega_above_one_duration": longest_duration(
            omega_above,
            dt,
        ),
        "longest_lambda_above_one_duration": longest_duration(
            lambda_above,
            dt,
        ),
    }


def save_numerical_outputs(
    t: np.ndarray,
    states: np.ndarray,
    derived: dict[str, np.ndarray],
    stats: dict[str, float | int | None],
) -> tuple[Path, Path]:
    """Save the full-frequency path and summary statistics."""
    DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)

    csv_path = DATA_DIRECTORY / f"{OUTPUT_STEM}_data.csv"
    json_path = DATA_DIRECTORY / f"{OUTPUT_STEM}_statistics.json"

    columns = np.column_stack(
        [
            t,
            states,
            derived["profit"],
            derived["debt_service"],
            derived["inflation"],
            derived["financing_gap"],
            derived["interest_growth_gap"],
            derived["outside_economic_region"].astype(int),
        ]
    )
    header = (
        "time,omega,lambda,ell,profit_share,debt_service,inflation,"
        "financing_gap,interest_growth_gap,outside_economic_region"
    )
    np.savetxt(
        csv_path,
        columns,
        delimiter=",",
        header=header,
        comments="",
    )

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2, sort_keys=True)

    return csv_path, json_path


# ============================================================
# Figure creation
# ============================================================
def create_baseline_figure(
    show: bool = True,
) -> tuple[plt.Figure, dict[str, float | int | None]]:
    """Generate and save the revised three-panel baseline figure."""
    model, t, states = simulate_baseline_path()
    derived = calculate_derived_variables(model, states)
    stats = summarize_baseline_path(t, states, derived)

    omega = states[:, 0]
    lam = states[:, 1]
    ell = states[:, 2]

    plot_indices = sample_indices(t, TIME_SERIES_INTERVAL)
    phase_indices = sample_indices(t, PHASE_INTERVAL)

    t_plot = t[plot_indices]
    omega_plot = omega[plot_indices]
    lam_plot = lam[plot_indices]
    ell_plot = ell[plot_indices]

    profit_plot = derived["profit"][plot_indices]
    debt_service_plot = derived["debt_service"][plot_indices]
    inflation_plot = derived["inflation"][plot_indices]

    t_phase = t[phase_indices]
    omega_phase = moving_average(
        omega[phase_indices],
        PHASE_SMOOTHING_WINDOW,
    )
    lambda_phase = moving_average(
        lam[phase_indices],
        PHASE_SMOOTHING_WINDOW,
    )
    ell_phase = moving_average(
        ell[phase_indices],
        PHASE_SMOOTHING_WINDOW,
    )

    outside_regions = contiguous_true_regions(
        derived["outside_economic_region"]
    )
    deflation_regions = contiguous_true_regions(derived["deflation"])

    fig = plt.figure(figsize=(16, 5.8))
    layout = gridspec.GridSpec(
        1,
        3,
        width_ratios=[1.05, 1.25, 1.0],
    )

    # --------------------------------------------------------
    # Panel (a): state variables
    # --------------------------------------------------------
    ax1 = fig.add_subplot(layout[0])

    ax1.plot(
        t_plot,
        omega_plot,
        color=COLOR_OMEGA,
        linewidth=1.5,
        label=r"$\omega_t$",
        zorder=3,
    )
    ax1.plot(
        t_plot,
        lam_plot,
        color=COLOR_LAMBDA,
        linewidth=1.5,
        label=r"$\lambda_t$",
        zorder=3,
    )
    ax1.plot(
        t_plot,
        ell_plot,
        color=COLOR_ELL,
        linewidth=1.5,
        label=r"$\ell_t$",
        zorder=3,
    )

    ax1.axhline(
        0.0,
        color=COLOR_REFERENCE,
        linestyle="--",
        linewidth=0.8,
        alpha=0.7,
    )
    ax1.axhline(
        1.0,
        color=COLOR_REFERENCE,
        linestyle="--",
        linewidth=0.8,
        alpha=0.7,
    )

    first_outside_label = True
    for start, end in outside_regions:
        ax1.axvspan(
            t[start],
            t[min(end, len(t) - 1)],
            facecolor=COLOR_OUTSIDE_REGION,
            edgecolor=COLOR_REFERENCE,
            alpha=0.10,
            hatch="//",
            label=(
                "Outside economic region"
                if first_outside_label
                else None
            ),
            zorder=0,
        )
        first_outside_label = False

    ax1.set_title("(a) State Variables", fontweight="bold")
    ax1.set_xlabel("Time (years)")
    ax1.set_ylabel("Value")
    ax1.set_xlim(0.0, T_SIM)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best", frameon=True)

    # --------------------------------------------------------
    # Panel (b): 3D phase trajectory
    # --------------------------------------------------------
    ax2 = fig.add_subplot(layout[1], projection="3d")

    points = np.column_stack(
        [omega_phase, lambda_phase, ell_phase]
    ).reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    normalization = plt.Normalize(
        float(t_phase.min()),
        float(t_phase.max()),
    )
    phase_line = Line3DCollection(
        segments,
        cmap=PHASE_CMAP,
        norm=normalization,
    )
    phase_line.set_array(t_phase[:-1])
    phase_line.set_linewidth(2.0)
    ax2.add_collection3d(phase_line)

    ax2.scatter(
        omega_phase[0],
        lambda_phase[0],
        ell_phase[0],
        s=55,
        marker="o",
        color=COLOR_START,
        label="Start",
        depthshade=False,
        zorder=4,
    )
    ax2.scatter(
        omega_phase[-1],
        lambda_phase[-1],
        ell_phase[-1],
        s=65,
        marker="X",
        color=COLOR_END,
        label="End",
        depthshade=False,
        zorder=4,
    )

    add_axis_padding(ax2, omega_phase, "x")
    add_axis_padding(ax2, lambda_phase, "y")
    add_axis_padding(ax2, ell_phase, "z")

    ax2.set_title("(b) Phase-Space Trajectory", fontweight="bold", pad=12)
    ax2.set_xlabel(r"Wage share $(\omega)$", labelpad=8)
    ax2.set_ylabel(r"Employment $(\lambda)$", labelpad=8)
    ax2.set_zlabel(r"Net debt $(\ell)$", labelpad=8)
    ax2.view_init(elev=28, azim=-58)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper left", frameon=True)

    colour_bar = fig.colorbar(
        phase_line,
        ax=ax2,
        ticks=np.linspace(0.0, T_SIM, 5),
        pad=0.08,
        shrink=0.8,
    )
    colour_bar.set_label("Time (years)")

    # --------------------------------------------------------
    # Panel (c): financial ratios
    # --------------------------------------------------------
    ax3 = fig.add_subplot(layout[2])

    ax3.plot(
        t_plot,
        debt_service_plot,
        color=COLOR_DEBT_SERVICE,
        linewidth=1.5,
        label=r"Debt service $(r\ell_t)$",
        zorder=3,
    )
    ax3.plot(
        t_plot,
        profit_plot,
        color=COLOR_PROFIT,
        linewidth=1.5,
        label=r"Profit share $(\pi_t)$",
        zorder=3,
    )
    ax3.plot(
        t_plot,
        inflation_plot,
        color=COLOR_INFLATION,
        linewidth=1.5,
        label=r"Inflation $(i(\omega_t))$",
        zorder=3,
    )
    ax3.axhline(
        0.0,
        color=COLOR_REFERENCE,
        linestyle="--",
        linewidth=0.8,
        alpha=0.7,
    )

    first_deflation_label = True
    for start, end in deflation_regions:
        ax3.axvspan(
            t[start],
            t[min(end, len(t) - 1)],
            facecolor=COLOR_DEFLATION_SHADE,
            edgecolor="none",
            alpha=0.12,
            label=(
                "Deflation episode"
                if first_deflation_label
                else None
            ),
            zorder=0,
        )
        first_deflation_label = False

    ax3.set_title("(c) Financial Ratios", fontweight="bold")
    ax3.set_xlabel("Time (years)")
    ax3.set_ylabel("Ratio")
    ax3.set_xlim(0.0, T_SIM)
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="best", frameon=True)

    fig.tight_layout()

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    png_path = OUTPUT_DIRECTORY / f"{OUTPUT_STEM}.png"
    pdf_path = OUTPUT_DIRECTORY / f"{OUTPUT_STEM}.pdf"

    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)

    csv_path, json_path = save_numerical_outputs(
        t,
        states,
        derived,
        stats,
    )

    print_summary(stats)
    print(f"\nSaved figure: {png_path}")
    print(f"Saved PDF:    {pdf_path}")
    print(f"Saved data:   {csv_path}")
    print(f"Saved stats:  {json_path}")

    if show:
        plt.show()

    return fig, stats


def format_optional_time(value: float | None) -> str:
    """Format an optional time value for console output."""
    return "not observed" if value is None else f"{value:.2f} years"


def print_summary(stats: dict[str, float | int | None]) -> None:
    """Print the diagnostics needed for the Section 5.2 rewrite."""
    print("\nBaseline positive-part stochastic simulation")
    print("=" * 54)
    print(f"Seed: {stats['seed']}")
    print(f"Horizon: {stats['T']:.2f} years")
    print(f"Time step: {stats['dt']:.4f} years")

    print("\nState-variable ranges")
    print("-" * 54)
    print(
        f"omega:  [{stats['omega_min']:.6f}, "
        f"{stats['omega_max']:.6f}]"
    )
    print(
        f"lambda: [{stats['lambda_min']:.6f}, "
        f"{stats['lambda_max']:.6f}]"
    )
    print(
        f"ell:    [{stats['ell_min']:.6f}, "
        f"{stats['ell_max']:.6f}]"
    )
    print(f"Mean ell: {stats['ell_mean']:.6f}")
    print(f"Std. ell: {stats['ell_std']:.6f}")
    print(
        "Net-savings share of horizon: "
        f"{100.0 * stats['net_savings_fraction']:.2f}%"
    )

    print("\nEconomic-region diagnostics")
    print("-" * 54)
    print(
        "First share-boundary exit: "
        f"{format_optional_time(stats['first_exit_time'])}"
    )
    print(
        "First omega > 1: "
        f"{format_optional_time(stats['omega_first_above_one'])}"
    )
    print(
        "First lambda > 1: "
        f"{format_optional_time(stats['lambda_first_above_one'])}"
    )
    print(
        "Fraction outside economic region: "
        f"{100.0 * stats['outside_economic_region_fraction']:.4f}%"
    )
    print(
        "Longest omega > 1 excursion: "
        f"{stats['longest_omega_above_one_duration']:.4f} years"
    )
    print(
        "Longest lambda > 1 excursion: "
        f"{stats['longest_lambda_above_one_duration']:.4f} years"
    )

    print("\nFinancial diagnostics")
    print("-" * 54)
    print(
        f"Profit share: [{stats['profit_min']:.6f}, "
        f"{stats['profit_max']:.6f}]"
    )
    print(
        f"Debt service: [{stats['debt_service_min']:.6f}, "
        f"{stats['debt_service_max']:.6f}]"
    )
    print(
        f"Inflation: [{stats['inflation_min']:.6f}, "
        f"{stats['inflation_max']:.6f}]"
    )
    print(
        "Deflation share of horizon: "
        f"{100.0 * stats['deflation_fraction']:.2f}%"
    )
    print(
        "Longest deflation episode: "
        f"{stats['longest_deflation_duration']:.4f} years"
    )


if __name__ == "__main__":
    create_baseline_figure(show=True)
